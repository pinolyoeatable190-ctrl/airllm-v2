# AirLLM Inference-Only Fork

Ce depot est recentre **strictement sur l'inference LLM haute performance**.

## Objectifs de ce fork

- Suppression des composants lies a l'entrainement (SFT, RLHF, datasets d'entrainement, scripts utilitaires hors inference).
- Conservation du coeur d'inference AirLLM (chargement par couches, faible VRAM, support multi-modeles).
- Optimisations runtime pour accelerer l'inference sur GPU.

## Installation minimale

```bash
pip install -r requirements.txt
pip install -e air_llm
```

## Exemple rapide

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    compression="4bit",  # ou "8bit"
)

prompt = ["Donne-moi 3 bonnes pratiques pour optimiser une inference LLM."]
inputs = model.tokenizer(prompt, return_tensors="pt", truncation=True)

out = model.generate(
    inputs["input_ids"].cuda(),
    max_new_tokens=128,
    use_cache=True,
)

print(model.tokenizer.decode(out[0], skip_special_tokens=True))
```

## Portee technique

Le projet inclut desormais uniquement ce qui est necessaire a :

- chargement et execution de modeles causaux ;
- compression 4/8 bits orientee inference ;
- prechargement de couches et execution efficace memoire/temps.

## Optimisations sans perte (lossless)

- **`layer_cache_size`** : conserve en RAM CPU les derniers shards de couches pour reduire les relectures disque (aucune perte de qualite).
- **Cache interne du masque causal et des `position_ids`** pour eviter les reconstructions tensorielles repetees.
- **Runtime tuning** : TF32, cuDNN benchmark, high-precision matmul actives automatiquement.
- **Import paresseux** de `optimum.BetterTransformer` : le module n'est charge que si disponible, evitant un crash d'import inutile.
- **Factorisation du code** : 5 fichiers de variantes triviales fusionnes en 1, imports simplifies.
- **Nettoyage notebooks** : outputs supprimes, duplications eliminees (exemples = liens vers tests).

Exemple:

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    layer_cache_size=4,  # cache CPU lossless des 4 dernieres couches
)
```


## Profil Raspberry Pi 5 (8GB, CPU + SD)

> AutoModel n'utilise plus de registry de modeles preselectionnes: tout repo Hugging Face compatible causal LM est tente via un chemin generique unique.


Pour un usage contraint (sans GPU), utilisez le profil preconfigure :

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    deployment_profile="rpi5_8gb_sd",
)
```

Ce profil applique automatiquement des valeurs orientees faible RAM/I/O SD : CPU mode, prefetch desactive, cache couche limite, nettoyage memoire plus espace, et barres de progression desactivees.

## Structure du repository

```
air_llm/
  airllm/
    airllm_base.py         # Moteur d'inference principal (sharded layer-by-layer)
    airllm_base.py         # Moteur d'inference universel (layer-wise)
    airllm_llama_mlx.py     # Backend MLX (macOS Apple Silicon)
    auto_model.py           # Auto-routage universel vers moteur generique
    utils.py                # Splitting, compression, I/O
    profiler.py             # Profiling par couche
    persist/                # Persistance (safetensors / MLX)
  examples/
  tests/
```


### Exemple RPi avec Qwen3.5

Qwen3.5 utilise une topologie interne `model.language_model.*` et est maintenant geree explicitement par l'auto-detection AirLLM.

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "Qwen/Qwen3.5-4B",
    deployment_profile="rpi5_8gb_sd",
    compression=None,
)
```
