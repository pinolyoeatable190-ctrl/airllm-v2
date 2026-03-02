from sys import platform

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .airllm_variants import (
        AirLLMLlama2,
        AirLLMMistral,
        AirLLMMixtral,
        AirLLMInternLM,
        AirLLMQWen2,
    )
    from .airllm_chatglm import AirLLMChatGLM
    from .airllm_qwen import AirLLMQWen
    from .airllm_baichuan import AirLLMBaichuan
    from .airllm_base import AirLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
