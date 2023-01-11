import flamb
from .base import LayerBase

class Linear(LayerBase):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = flamb.rand((input_size, output_size), requires_grad=True)
        self.bias = flamb.rand((output_size,))

    def forward(self, x):
        assert (x.shape[-1] == self.input_size), f"Input size of x should be {self.input_size}, but got {x.shape[-1]}"
        x = x.dot(self.weights)
        x = x + self.bias
        return x

    def get_parameters(self):
        return flamb.concatenate(self.weights.flatten(), self.bias)
