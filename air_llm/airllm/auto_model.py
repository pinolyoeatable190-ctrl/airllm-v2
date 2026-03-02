import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx

# Architecture -> (module_path, class_name) mapping
# All trivial variants live in airllm_variants; specialised ones in their own files.
_ARCH_REGISTRY = {
    "Qwen2ForCausalLM": ("airllm.airllm_variants", "AirLLMQWen2"),
    "QWen":             ("airllm.airllm_qwen",     "AirLLMQWen"),
    "Baichuan":         ("airllm.airllm_baichuan",  "AirLLMBaichuan"),
    "ChatGLM":          ("airllm.airllm_chatglm",   "AirLLMChatGLM"),
    "InternLM":         ("airllm.airllm_variants",  "AirLLMInternLM"),
    "Mistral":          ("airllm.airllm_variants",  "AirLLMMistral"),
    "Mixtral":          ("airllm.airllm_variants",  "AirLLMMixtral"),
    "Llama":            ("airllm.airllm_variants",  "AirLLMLlama2"),
}

_DEFAULT_MODULE = "airllm.airllm_variants"
_DEFAULT_CLASS  = "AirLLMLlama2"

def _apply_profile_defaults(kwargs):
    """Apply optional deployment profiles without overriding explicit user args."""
    profile = kwargs.pop("deployment_profile", None)
    if profile is None:
        return kwargs

    if profile == "rpi5_8gb_sd":
        defaults = {
            "device": "cpu",
            "dtype": None,
            "prefetching": False,
            "prefetch_window": 1,
            "layer_cache_size": 1,
            "cleanup_interval": 16,
            "cleanup_memory_pressure": 0.97,
            "rebuild_model_per_forward": False,
            "cpu_thread_count": 4,
            "cpu_interop_threads": 1,
            "disable_progress_bar": True,
        }
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        return kwargs

    raise ValueError(f"Unknown deployment_profile: {profile}")



class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        token_kwargs = {"token": kwargs["hf_token"]} if "hf_token" in kwargs else {}
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **token_kwargs
        )

        arch = config.architectures[0] if config.architectures else ""
        for key, (mod, klass) in _ARCH_REGISTRY.items():
            if key in arch:
                return mod, klass

        print(f"unknown architecture: {arch}, falling back to Llama2...")
        return _DEFAULT_MODULE, _DEFAULT_CLASS

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        kwargs = _apply_profile_defaults(kwargs)

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module_name, cls_name = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module_name)
        class_ = getattr(module, cls_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
