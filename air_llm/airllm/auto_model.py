import inspect
from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx


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


def _filter_supported_init_kwargs(class_, kwargs):
    try:
        sig = inspect.signature(class_.__init__)
    except (TypeError, ValueError):
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    supported = {
        name for name, p in sig.parameters.items()
        if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if k in supported}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        deployment_profile = kwargs.pop("deployment_profile", None)
        if deployment_profile is not None:
            kwargs["deployment_profile"] = deployment_profile
        kwargs = _apply_profile_defaults(kwargs)

        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        from .airllm_base import AirLLMBaseModel
        filtered_kwargs = _filter_supported_init_kwargs(AirLLMBaseModel, kwargs)
        return AirLLMBaseModel(pretrained_model_name_or_path, *inputs, **filtered_kwargs)
