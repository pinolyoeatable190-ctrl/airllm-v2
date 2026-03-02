"""Compact model variants that only differ from the base by simple overrides.

Models with unique logic (QWen, ChatGLM, Baichuan, MLX) keep their own files.
Models that are trivial subclasses are all defined here to avoid file sprawl.
"""

from transformers import GenerationConfig
from .airllm_base import AirLLMBaseModel


class AirLLMLlama2(AirLLMBaseModel):
    """Llama-2 / Llama-3 / generic Llama-like models (default behaviour)."""
    pass


class AirLLMMistral(AirLLMBaseModel):
    """Mistral models - disable BetterTransformer, use empty GenerationConfig."""

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()


class AirLLMMixtral(AirLLMBaseModel):
    """Mixtral MoE models."""

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()


class AirLLMInternLM(AirLLMBaseModel):
    """InternLM models."""

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()


class AirLLMQWen2(AirLLMBaseModel):
    """QWen2 models (uses standard layer names, just disables BetterTransformer)."""

    def get_use_better_transformer(self):
        return False
