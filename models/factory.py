from models.qwen_engine import Qwen2_5_VLEngine
from models.minicpm_engine import MiniCPMEngine

SUPPORTED_MODELS = {
    "qwen2.5-vl": Qwen2_5_VLEngine,
    "minicpm":    MiniCPMEngine,
}

def create_engine(config):
    model_type = config.model.model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_type}' is not supported. "
            f"Choose: {list(SUPPORTED_MODELS.keys())}"
        )
    return SUPPORTED_MODELS[model_type](config)