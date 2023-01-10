import flamb
from .base import LayerBase

class Linear(LayerBase):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.tensor = flamb.zeros((input_size, output_size), requires_grad=True)
