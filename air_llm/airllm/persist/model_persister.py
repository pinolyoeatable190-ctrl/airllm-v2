"""Abstract model persistence layer with platform-aware factory."""

from sys import platform

_model_persister = None


class ModelPersister:
    @classmethod
    def get_model_persister(cls):
        global _model_persister
        if _model_persister is not None:
            return _model_persister

        if platform == "darwin":
            from .mlx_model_persister import MlxModelPersister
            _model_persister = MlxModelPersister()
        else:
            from .safetensor_model_persister import SafetensorModelPersister
            _model_persister = SafetensorModelPersister()
        return _model_persister

    def model_persist_exist(self, layer_name, saving_path):
        raise NotImplementedError

    def persist_model(self, state_dict, layer_name, path):
        raise NotImplementedError

    def load_model(self, layer_name, path):
        raise NotImplementedError
