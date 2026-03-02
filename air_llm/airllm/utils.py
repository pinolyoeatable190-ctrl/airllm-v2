"""Utility functions for AirLLM: memory management, layer splitting, compression."""

import gc
import json
import os
import ctypes
import shutil
import importlib.util
from tqdm import tqdm
from pathlib import Path
from glob import glob
import time
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from sys import platform

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from .persist import ModelPersister

is_on_mac_os = platform == "darwin"

bnb = None
bitsandbytes_installed = importlib.util.find_spec("bitsandbytes") is not None


def _get_bnb():
    global bnb
    if bnb is None:
        import bitsandbytes as _bnb
        bnb = _bnb
    return bnb

import huggingface_hub


def save_quant_state_to_dict(self, packed=True):
    """Serialize bitsandbytes QuantState to a dict of tensors.

    Replacement for bnb quantstat.as_dict(True) until the upstream bug is fixed.
    """
    qs_dict = {
        'quant_type': self.quant_type,
        'absmax': self.absmax,
        'blocksize': self.blocksize,
        'quant_map': self.code,
        'dtype': str(self.dtype).strip('torch.'),
        'shape': tuple(self.shape),
    }
    if self.nested:
        qs_dict.update({
            'nested_absmax': self.state2.absmax,
            'nested_blocksize': self.state2.blocksize,
            'nested_quant_map': self.state2.code,
            'nested_dtype': str(self.state2.dtype).strip('torch.'),
            'nested_offset': self.offset.item(),
        })
    if not packed:
        return qs_dict

    qs_packed = {k: v for k, v in qs_dict.items() if isinstance(v, torch.Tensor)}
    non_tensor = {k: v for k, v in qs_dict.items() if not isinstance(v, torch.Tensor)}
    bnb_mod = _get_bnb()
    qs_packed[f"quant_state.bitsandbytes__{self.quant_type}"] = bnb_mod.utils.pack_dict_to_tensor(non_tensor)
    return qs_packed


class NotEnoughSpaceException(Exception):
    pass


def clean_memory():
    """Free CPU RAM and GPU VRAM."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def uncompress_layer_state_dict(layer_state_dict):
    """Decompress a 4-bit or 8-bit quantized layer state dict back to float16."""
    keys = list(layer_state_dict.keys())

    if any('4bit' in k for k in keys):
        out = {}
        for k, v in layer_state_dict.items():
            if '4bit' not in k:
                qs = {kk[len(k):]: kv for kk, kv in layer_state_dict.items() if kk.startswith(k) and k != kk}
                bnb_mod = _get_bnb()
                quant_state = bnb_mod.functional.QuantState.from_dict(qs_dict=qs, device="cuda")
                out[k] = bnb_mod.functional.dequantize_nf4(v.cuda(), quant_state)
        return out

    if any('8bit' in k for k in keys):
        out = {}
        for k, v in layer_state_dict.items():
            if '8bit' not in k:
                bnb_mod = _get_bnb()
                out[k] = bnb_mod.functional.dequantize_blockwise(
                    v.cuda(),
                    bnb_mod.functional.QuantState(
                        absmax=layer_state_dict[k + ".8bit.absmax"].cuda(),
                        code=layer_state_dict[k + ".8bit.code"].cuda(),
                        blocksize=2048,
                        dtype=torch.float16,
                    ),
                )
        return out

    return layer_state_dict


def load_layer(local_path, layer_name, profiling=False):
    """Load a single layer from persisted storage and optionally decompress."""
    layer_state_dict = ModelPersister.get_model_persister().load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    result = uncompress_layer_state_dict(layer_state_dict)

    if profiling:
        return result, time.process_time() - t
    return result


def check_space(checkpoint_path, layer_shards_saving_path=None, compression=None,
                splitted_model_dir_name='splitted_model'):
    total_shard_bytes = sum(
        os.path.getsize(f) for f in glob(str(checkpoint_path / '*'))
    )

    existing_split_bytes = 0
    if layer_shards_saving_path is not None:
        existing_split_bytes = sum(
            os.path.getsize(f)
            for f in glob(str(Path(layer_shards_saving_path) / splitted_model_dir_name / '*'))
        )

    if compression == '4bit':
        total_shard_bytes = int(total_shard_bytes / 0.2813)
    elif compression == '8bit':
        total_shard_bytes //= 2

    target = checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    _, _, free = shutil.disk_usage(target)

    if free + existing_split_bytes < total_shard_bytes:
        raise NotEnoughSpaceException(
            f"Not enough space. Free: {free / 2**30:.2f}GB. "
            f"Model: {total_shard_bytes / 2**30:.2f}GB. "
            f"Reusable: {existing_split_bytes / 2**30:.2f}GB."
        )


def compress_layer_state_dict(layer_state_dict, compression=None):
    """Quantize a layer state dict to 4-bit or 8-bit."""
    if compression == '4bit':
        out = {}
        bnb_mod = _get_bnb()
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb_mod.functional.quantize_nf4(v.cuda(), blocksize=64)
            out[k] = v_quant
            for qs_k, qs_v in save_quant_state_to_dict(quant_state).items():
                out[f"{k}.4bit.{qs_k}"] = qs_v
        return out

    if compression == '8bit':
        out = {}
        bnb_mod = _get_bnb()
        for k, v in layer_state_dict.items():
            v_quant, quant_state = bnb_mod.functional.quantize_blockwise(v.cuda(), blocksize=2048)
            out[k] = v_quant
            out[f"{k}.8bit.absmax"] = quant_state.absmax.clone().contiguous()
            out[f"{k}.8bit.code"] = quant_state.code.clone().contiguous()
        return out

    return layer_state_dict


def _remove_file_and_target(path):
    """Remove a file and its symlink target if applicable."""
    real = os.path.realpath(path)
    os.remove(path)
    if real != path and os.path.exists(real):
        os.remove(real)


def split_and_save_layers(checkpoint_path, layer_shards_saving_path=None,
                          splitted_model_dir_name='splitted_model',
                          compression=None, layer_names=None,
                          delete_original=False, repo_id=None, hf_token=None,
                          split_cleanup_interval=8):
    """Split a sharded model checkpoint into per-layer safetensor files."""
    if compression is not None:
        assert bitsandbytes_installed, "bitsandbytes is required for compression."
        splitted_model_dir_name = f"{splitted_model_dir_name}.{compression}"

    checkpoint_path = Path(checkpoint_path)
    split_cleanup_interval = max(1, int(split_cleanup_interval))
    saving_path = (
        Path(layer_shards_saving_path) / splitted_model_dir_name
        if layer_shards_saving_path
        else checkpoint_path / splitted_model_dir_name
    )

    # Determine format
    safetensors_format = not os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json')
    if safetensors_format:
        index_file = 'model.safetensors.index.json'
        assert os.path.exists(checkpoint_path / index_file), f'{index_file} should exist.'
    else:
        index_file = 'pytorch_model.bin.index.json'

    with open(checkpoint_path / index_file, 'rb') as f:
        index = json.load(f)['weight_map']

    # Count layers (robust across variants such as model.layers.0.* and model.layers.layers.0.*)
    prefix = layer_names['layer_prefix'] if layer_names else 'model.layers'
    layer_ids = set()
    layer_infix_counts = defaultdict(int)
    for k in index:
        if prefix not in k:
            continue

        if layer_names:
            if not k.startswith(prefix):
                continue
            remainder = k[len(prefix):].lstrip('.')
            parts = [p for p in remainder.split('.') if p]
            layer_id = None
            for j, part in enumerate(parts):
                if part.isdigit():
                    layer_id = int(part)
                    layer_infix_counts['.'.join(parts[:j])] += 1
                    break
        else:
            parts = k.split('.')
            layer_id = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None

        if layer_id is not None:
            layer_ids.add(layer_id)

    if not layer_ids:
        sample = next(iter(index.keys())) if index else '<empty-index>'
        raise ValueError(
            f"Unable to infer transformer layer ids for prefix '{prefix}'. Sample key: {sample}"
        )

    n_layers = max(layer_ids) + 1
    layer_infix = ''
    if layer_names and layer_infix_counts:
        layer_infix = max(layer_infix_counts.items(), key=lambda x: x[1])[0]

    # Build layer list
    if layer_names is None:
        layers = ['model.embed_tokens.'] + [f'model.layers.{i}.' for i in range(n_layers)] + ['model.norm.', 'lm_head.']
    else:
        infix = f'.{layer_infix}' if layer_infix else ''
        layers = [layer_names['embed']] + [f'{layer_names["layer_prefix"]}{infix}.{i}' for i in range(n_layers)] + [layer_names['norm'], layer_names['lm_head']]
        if 'rotary_pos_emb' in layer_names:
            layers = [layer_names['rotary_pos_emb']] + layers
        layers = [l + "." for l in layers]

    # Check if already split
    if os.path.exists(saving_path):
        found = {l: ModelPersister.get_model_persister().model_persist_exist(l, saving_path) for l in layers}
        if all(found.values()):
            return str(saving_path)

    if not delete_original:
        check_space(checkpoint_path, layer_shards_saving_path, compression, splitted_model_dir_name)

    saving_path.mkdir(parents=True, exist_ok=True)

    # Precompute which tensors/files are needed per layer.
    layer_to_file_keys = {}
    for layer in layers:
        file_keys = defaultdict(list)
        for key, filename in index.items():
            if key.startswith(layer):
                file_keys[filename].append(key)
        layer_to_file_keys[layer] = dict(file_keys)

    # Deletion safety: only remove original files once every dependent layer is persisted.
    file_refcount = defaultdict(int)
    for file_keys in layer_to_file_keys.values():
        for f in file_keys.keys():
            file_refcount[f] += 1

    last_loaded_bin = None
    last_bin_state_dict = None

    for ilayer, layer in enumerate(tqdm(layers)):
        file_keys = layer_to_file_keys[layer]
        layer_sd = {}

        for filename, needed_keys in file_keys.items():
            to_load = checkpoint_path / filename
            if not os.path.exists(to_load):
                assert repo_id is not None
                huggingface_hub.snapshot_download(repo_id, allow_patterns=os.path.basename(to_load), token=hf_token)

            if safetensors_format:
                with safe_open(str(to_load), framework='pt', device='cpu') as sf:
                    for k in needed_keys:
                        layer_sd[k] = sf.get_tensor(k)
            else:
                if last_loaded_bin != filename:
                    last_bin_state_dict = torch.load(to_load, map_location='cpu')
                    last_loaded_bin = filename
                for k in needed_keys:
                    layer_sd[k] = last_bin_state_dict[k]

        layer_sd = compress_layer_state_dict(layer_sd, compression)

        if not ModelPersister.get_model_persister().model_persist_exist(layer, saving_path):
            ModelPersister.get_model_persister().persist_model(layer_sd, layer, saving_path)

        if delete_original:
            for filename in file_keys.keys():
                file_refcount[filename] -= 1
                if file_refcount[filename] == 0:
                    _remove_file_and_target(checkpoint_path / filename)
                    if last_loaded_bin == filename:
                        last_loaded_bin = None
                        last_bin_state_dict = None

        del layer_sd
        if ((ilayer + 1) % split_cleanup_interval == 0) or (ilayer + 1 == len(layers)):
            clean_memory()

    return str(saving_path)




def infer_layer_layout_and_checkpoint(model_local_path_or_repo_id, hf_token=None):
    """Infer layer naming layout and return resolved local checkpoint path.

    Returns:
        tuple(Path, dict): (local_checkpoint_path, layer_names_dict)
    """
    p = Path(model_local_path_or_repo_id)
    if os.path.exists(p):
        checkpoint_path = p
    else:
        checkpoint_path = Path(huggingface_hub.snapshot_download(
            model_local_path_or_repo_id,
            token=hf_token,
            ignore_patterns=['*.safetensors', '*.bin'],
        ))

    if os.path.exists(checkpoint_path / 'model.safetensors.index.json'):
        index_file = checkpoint_path / 'model.safetensors.index.json'
    elif os.path.exists(checkpoint_path / 'pytorch_model.bin.index.json'):
        index_file = checkpoint_path / 'pytorch_model.bin.index.json'
    else:
        return checkpoint_path, {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }

    with open(index_file, 'rb') as f:
        weight_map = json.load(f)['weight_map']

    keys = list(weight_map.keys())

    if any(k.startswith('model.language_model.layers.') for k in keys):
        embed = 'model.language_model.embed_tokens'
        layer_prefix = 'model.language_model.layers'
        norm = 'model.language_model.norm'
    elif any(k.startswith('transformer.h.') for k in keys):
        embed = 'transformer.wte'
        layer_prefix = 'transformer.h'
        norm = 'transformer.ln_f'
    else:
        embed = 'model.embed_tokens'
        layer_prefix = 'model.layers'
        norm = 'model.norm'

    lm_head = 'lm_head' if any(k.startswith('lm_head.') for k in keys) else 'model.lm_head'

    return checkpoint_path, {
        'embed': embed,
        'layer_prefix': layer_prefix,
        'norm': norm,
        'lm_head': lm_head,
    }


def infer_layer_names_dict(model_local_path_or_repo_id, hf_token=None):
    """Backward-compatible wrapper returning only layer names."""
    _, layer_names = infer_layer_layout_and_checkpoint(model_local_path_or_repo_id, hf_token=hf_token)
    return layer_names

def find_or_create_local_splitted_path(model_local_path_or_repo_id, layer_shards_saving_path=None,
                                       compression=None, layer_names=None, hf_token=None,
                                       delete_original=False):
    """Find local model cache, download if needed, then split into per-layer files."""
    p = Path(model_local_path_or_repo_id)
    if os.path.exists(p):
        if os.path.exists(p / 'pytorch_model.bin.index.json') or os.path.exists(p / 'model.safetensors.index.json'):
            return p, split_and_save_layers(
                p, layer_shards_saving_path,
                compression=compression, layer_names=layer_names,
                delete_original=delete_original,
            )

    hf_cache_path = huggingface_hub.snapshot_download(
        model_local_path_or_repo_id, token=hf_token,
        ignore_patterns=['*.safetensors', '*.bin'],
    )

    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path, layer_shards_saving_path,
        compression=compression, layer_names=layer_names,
        delete_original=delete_original,
        repo_id=model_local_path_or_repo_id, hf_token=hf_token,
    )
