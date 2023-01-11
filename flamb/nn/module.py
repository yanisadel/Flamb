import flamb
from .layers.base import LayerBase

class Module:
    def __init__(self):
        self.parameters = None

    def initialize_parameters(self):
        self.parameters = flamb.to_tensor([])
        for param in self.__dict__.values():
            if isinstance(param, LayerBase):
                self.parameters = flamb.concatenate(self.parameters, param.get_parameters())

    def __call__(self, x):
        raise Exception("You need to implement the __call__ method")
