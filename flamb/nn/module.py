import flamb
from .layers.base import LayerBase

class Module:
    def __init__(self):
        self.parameters = None

    def __call__(self, x): # This way of initializing parameters may cause a problem if we want to define an optimizer on model parameters, before calling the model
        if self.parameters is None:
            self.initialize_parameters()
        
        return self.forward(x)

    def initialize_parameters(self):
        self.parameters = flamb.to_tensor([])
        for param in self.__dict__.values():
            if isinstance(param, LayerBase):
                self.parameters = flamb.concatenate(self.parameters, param.get_parameters())

    def forward(self, x):
        raise Exception("You need to implement the forward method")
