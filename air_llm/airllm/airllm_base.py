"""Core sharded inference engine for AirLLM.

Loads model weights layer-by-layer to keep GPU memory bounded, regardless of
total model size.  This file contains the base class that all model variants
inherit from.
"""

from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import torch
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    GenerationMixin, GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer

from .profiler import LayeredProfiler
from .utils import clean_memory, load_layer, find_or_create_local_splitted_path

try:
    from optimum.bettertransformer import BetterTransformer
    _bettertransformer_available = True
except ImportError:
    _bettertransformer_available = False

try:
    import bitsandbytes as bnb
    bitsandbytes_installed = True
except ImportError:
    bitsandbytes_installed = False

try:
    from transformers.cache_utils import Cache
    cache_utils_installed = True
except ImportError:
    cache_utils_installed = False

# ---------------------------------------------------------------------------
# Inference-focused runtime tuning (called once at module load)
# ---------------------------------------------------------------------------
_RUNTIME_CONFIGURED = False

def _configure_runtime_once():
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    _RUNTIME_CONFIGURED = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _resolve_nested_attr(obj, dotted_name: str):
    """Resolve 'a.b.c' on *obj* to obj.a.b.c."""
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


class AirLLMBaseModel(GenerationMixin):
    """Sharded layer-wise inference engine.

    The model is split into per-layer safetensor files.  During forward, each
    layer is loaded from disk -> GPU, executed, then off-loaded to 'meta' to
    free VRAM.  An optional LRU CPU cache (`layer_cache_size`) avoids redundant
    disk reads for the most recently used layers.
    """

    # -- Override points for model variants ---------------------------------

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }

    def get_use_better_transformer(self):
        return True

    def get_generation_config(self):
        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception:
            return GenerationConfig()

    def get_tokenizer(self, hf_token=None):
        kw = {"token": hf_token} if hf_token else {}
        return AutoTokenizer.from_pretrained(
            self.model_local_path, trust_remote_code=True, **kw
        )

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[2]

    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'past_key_value': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}

    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    # -- Construction -------------------------------------------------------

    def __init__(
        self,
        model_local_path_or_repo_id,
        device="cuda:0",
        dtype=torch.float16,
        max_seq_len=512,
        layer_shards_saving_path=None,
        profiling_mode=False,
        compression=None,
        hf_token=None,
        prefetching=True,
        delete_original=False,
        layer_cache_size=2,
    ):
        _configure_runtime_once()

        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()
        self._supports_cache_class = False
        self.hf_quantizer = None
        self.layer_cache_size = max(0, int(layer_cache_size))
        self.layer_cache = OrderedDict()
        self._cached_attention_mask = None
        self._cached_position_ids = None

        if compression is not None and not bitsandbytes_installed:
            raise ImportError(
                "bitsandbytes is required for compression. "
                "Install it with: pip install bitsandbytes"
            )

        self.compression = compression
        self.hf_token = hf_token
        self.set_layer_names_dict()

        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(
            model_local_path_or_repo_id,
            layer_shards_saving_path,
            compression=compression,
            layer_names=self.layer_names_dict,
            hf_token=hf_token,
            delete_original=delete_original,
        )

        self.running_device = device
        self.device = torch.device(device)
        self.running_dtype = dtype
        self.dtype = dtype

        kw = {"token": hf_token} if hf_token else {}
        self.config = AutoConfig.from_pretrained(
            self.model_local_path, trust_remote_code=True, **kw
        )
        self.generation_config = self.get_generation_config()
        self.tokenizer = self.get_tokenizer(hf_token=hf_token)

        self.init_model()

        # Build ordered layer name list
        layers_count = len(_resolve_nested_attr(self.model, self.layer_names_dict["layer_prefix"]))
        lnd = self.layer_names_dict
        self.layer_names = (
            [lnd['embed']]
            + [f'{lnd["layer_prefix"]}.{i}' for i in range(layers_count)]
            + [lnd['norm'], lnd['lm_head']]
        )

        self.max_seq_len = max_seq_len
        self.main_input_name = "input_ids"

        # Prefetching config
        self.prefetching = prefetching
        if self.compression is not None:
            self.prefetching = False

        self.stream = (
            torch.cuda.Stream()
            if self.prefetching and device.startswith("cuda")
            else None
        )

    # -- Model initialisation -----------------------------------------------

    def init_model(self):
        self.model = None

        if self.get_use_better_transformer() and _bettertransformer_available:
            # Try BetterTransformer first
            try:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                    self.model = BetterTransformer.transform(self.model)
            except (ValueError, Exception):
                del self.model
                clean_memory()
                self.model = None

            # Fallback: SDPA attention
            if self.model is None:
                try:
                    self.config.attn_implementation = "sdpa"
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(
                            self.config, attn_implementation="sdpa", trust_remote_code=True
                        )
                except (TypeError, Exception):
                    del self.model
                    clean_memory()
                    self.model = None

        # Ultimate fallback
        if self.model is None:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()
        self._set_layers_from_names()

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device,
                value=buffer, dtype=self.running_dtype,
            )

        if 'rotary_pos_emb' in self.layer_names_dict:
            state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
            self.move_layer_to_device(state_dict)

    def _set_layers_from_names(self):
        lnd = self.layer_names_dict
        self.layers = [
            _resolve_nested_attr(self.model, lnd["embed"]),
            *list(_resolve_nested_attr(self.model, lnd["layer_prefix"])),
            _resolve_nested_attr(self.model, lnd["norm"]),
            _resolve_nested_attr(self.model, lnd["lm_head"]),
        ]

    # -- Layer cache (LRU, CPU-side) ----------------------------------------

    def _layer_cache_get(self, layer_name):
        if self.layer_cache_size <= 0:
            return None
        sd = self.layer_cache.get(layer_name)
        if sd is not None:
            self.layer_cache.move_to_end(layer_name)
        return sd

    def _layer_cache_put(self, layer_name, state_dict):
        if self.layer_cache_size <= 0:
            return
        self.layer_cache[layer_name] = state_dict
        self.layer_cache.move_to_end(layer_name)
        while len(self.layer_cache) > self.layer_cache_size:
            self.layer_cache.popitem(last=False)

    # -- Cached causal mask / position ids ----------------------------------

    def _get_full_attention_mask_and_position_ids(self):
        if self._cached_attention_mask is None or self._cached_position_ids is None:
            m = torch.ones(self.max_seq_len, self.max_seq_len, device=self.running_device)
            self._cached_attention_mask = m.triu(diagonal=1)[None, None, ...] == 0
            self._cached_position_ids = torch.arange(
                self.max_seq_len, dtype=torch.long, device=self.running_device
            )[None, :]
        return self._cached_attention_mask, self._cached_position_ids

    # -- Layer I/O ----------------------------------------------------------

    def load_layer_to_cpu(self, layer_name):
        cached = self._layer_cache_get(layer_name)
        if cached is not None:
            if self.profiling_mode:
                self.profiler.add_profiling_time('load_safe_tensor', 0.0)
            return cached

        t = time.time()
        result = load_layer(self.checkpoint_path, layer_name, self.profiling_mode)
        elapsed = time.time() - t

        if self.profiling_mode:
            state_dict, comp_time = result
            self.profiler.add_profiling_time('load_safe_tensor', elapsed - comp_time)
            self.profiler.add_profiling_time('compression_time', comp_time)
        else:
            state_dict = result

        if self.prefetching and torch.cuda.is_available():
            t = time.time()
            for v in state_dict.values():
                v.pin_memory()
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', time.time() - t)

        self._layer_cache_put(layer_name, state_dict)
        return state_dict

    def move_layer_to_device(self, state_dict):
        if self.hf_quantizer is None:
            param_names = list(state_dict.keys())
        else:
            seen = set()
            param_names = []
            for pn in state_dict:
                if '.weight' in pn:
                    ln = pn[:pn.index(".weight") + len(".weight")]
                    if ln not in seen:
                        seen.add(ln)
                        param_names.append(ln)

        for pn in param_names:
            if self.hf_quantizer is None or not self.hf_quantizer.check_quantized_param(
                self.model, param_value=None, param_name=pn, state_dict={}
            ):
                set_module_tensor_to_device(
                    self.model, pn, self.running_device,
                    value=state_dict[pn], dtype=self.running_dtype,
                )
            else:
                self.hf_quantizer.create_quantized_param(
                    self.model, state_dict[pn], pn, self.running_device, state_dict
                )
        return param_names

    # -- GenerationMixin interface ------------------------------------------

    def can_generate(self):
        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs,
    ):
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values)
            remove = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove:]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    # -- Forward pass -------------------------------------------------------

    def _load_and_move(self, layer_name, executor, future_ref):
        """Load a layer's weights (prefetch-aware) and move them to device.

        Returns (state_dict, moved_param_names).
        When prefetching is on, *future_ref* carries the already-submitted
        future.  Otherwise we load synchronously.
        """
        if self.prefetching:
            if self.profiling_mode:
                t = time.time()
            state_dict = future_ref.result()
            if self.profiling_mode:
                self.profiler.add_profiling_time('load_safe_tensor_cpu_wait', time.time() - t)
        else:
            state_dict = self.load_layer_to_cpu(layer_name)

        if self.profiling_mode:
            t = time.time()
        moved = self.move_layer_to_device(state_dict)
        if self.profiling_mode:
            self.profiler.add_profiling_time('create_layer_from_state_dict', time.time() - t)
        return moved

    def _run_transformer_layer(self, layer, seq, i, attention_mask, position_ids,
                               past_key_values, use_cache, kv_cache_list,
                               output_attentions, all_self_attns):
        """Execute a single transformer decoder layer on *seq*."""
        if past_key_values is not None:
            k_cache, v_cache = past_key_values[i - 1]
            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
            len_s = self.get_sequence_len(seq)

            kwargs = {
                'use_cache': True,
                **self.get_past_key_value_args(k_cache, v_cache),
                **self.get_pos_emb_args(len_p, len_s),
                **self.get_attention_mask_args(attention_mask, len_p, len_s),
                **self.get_position_ids_args(position_ids, len_p, len_s),
            }
            layer_outputs = layer(seq, **kwargs)
            new_seq = layer_outputs[0]

            if output_attentions:
                all_self_attns[i].append(layer_outputs[1])
            if use_cache:
                kv = layer_outputs[2 if output_attentions else 1]
                kv_cache_list[i][0].append(kv[0])
                kv_cache_list[i][1].append(kv[1])
        else:
            len_seq = self.get_sequence_len(seq)
            kwargs = {
                'use_cache': bool(use_cache),
                **self.get_pos_emb_args(0, len_seq),
                **self.get_attention_mask_args(attention_mask, 0, len_seq),
                **self.get_position_ids_args(position_ids, 0, len_seq),
            }
            layer_out = layer(seq, **kwargs)

            if use_cache:
                new_seq = layer_out[0]
                kv = layer_out[1] if not isinstance(layer_out[1], tuple) else layer_out[1]
                # Handle both (new_seq, (k,v)) and (new_seq, k, v) formats
                if isinstance(layer_out, tuple) and len(layer_out) >= 2:
                    new_seq = layer_out[0]
                    kv_pair = layer_out[1]
                    if isinstance(kv_pair, tuple) and len(kv_pair) == 2:
                        kv_cache_list[i][0].append(kv_pair[0])
                        kv_cache_list[i][1].append(kv_pair[1])
            else:
                new_seq = layer_out[0]

        return new_seq

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if cache_utils_installed and isinstance(past_key_values, Cache):
            use_cache = False
            past_key_values = None

        if self.profiling_mode:
            self.profiler.clear_profiling_time()
            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Re-initialise model so buffers are clean
        del self.model
        clean_memory()
        self.init_model()

        batch = [uid.to(self.running_device, non_blocking=True).unsqueeze(0) for uid in input_ids]

        attention_mask, position_ids = self._get_full_attention_mask_and_position_ids()

        kv_cache_list = [([], []) for _ in self.layers] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None
        all_self_attns = [[] for _ in self.layers] if output_attentions else None

        lnd = self.layer_names_dict

        with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            if self.prefetching:
                future = executor.submit(self.load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f'running layers({self.running_device})',
                total=len(self.layers),
            ):
                moved = self._load_and_move(layer_name, executor, future)

                # Kick off next layer prefetch
                if self.prefetching and (i + 1) < len(self.layer_names):
                    if self.profiling_mode:
                        t = time.time()
                    future = executor.submit(self.load_layer_to_cpu, self.layer_names[i + 1])
                    if self.profiling_mode:
                        self.profiler.add_profiling_time('kick_off_load_cpu', time.time() - t)

                # Execute layer on each sequence in the batch
                for j, seq in enumerate(batch):
                    if layer_name == lnd['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == lnd['norm']:
                        batch[j] = self.run_norm(layer, seq)
                    elif layer_name == lnd['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        batch[j] = self._run_transformer_layer(
                            layer, seq, i, attention_mask, position_ids,
                            past_key_values, use_cache, kv_cache_list,
                            output_attentions, all_self_attns,
                        )

                if output_hidden_states:
                    all_hidden_states.append(torch.cat(batch, 0))

                # Off-load layer to free VRAM
                if self.hf_quantizer is not None:
                    for pn in moved:
                        set_module_tensor_to_device(self.model, pn, 'meta')
                layer.to("meta")
                clean_memory()

        logits = torch.cat(batch, 0)

        if use_cache:
            kv_cache_list = [
                (torch.cat(kv[0], 0), torch.cat(kv[1], 0))
                for kv in kv_cache_list[1:-2]
            ]

        if output_attentions:
            all_self_attns = [
                torch.cat(a, 0) for a in all_self_attns[:-2]
            ]

        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-2]

        if not return_dict:
            return tuple(
                v for v in [
                    logits,
                    tuple(kv_cache_list) if kv_cache_list is not None else None,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_self_attns) if all_self_attns is not None else None,
                ] if v is not None
            )

        if self.profiling_mode:
            self.profiler.print_profiling_time()
            print(f"total infer process time: {time.process_time() - forward_start:.04f}")
            print(f"total infer wall time: {time.time() - forward_start_wall:.04f}")
            self.profiler.clear_profiling_time()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_self_attns is not None else None,
        )
