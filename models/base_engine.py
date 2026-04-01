# base_engine.py
from abc import ABC, abstractmethod

class BaseVLMEngine(ABC):
    def __init__(self, config):
        self.config = config
        self.processor = None
        self.model = None

    @abstractmethod
    def load_model_for_training(self):
        pass

    @abstractmethod
    def load_model_for_inference(self):
        pass

    def get_model(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return self.model

    def get_processor(self):
        if self.processor is None:
            raise RuntimeError("Processor is not loaded")
        return self.processor