import flamb
from .utils import *
import numpy as np

class Tensor(np.ndarray):
    # For the moment, Tensor can only contain flamb.Variable values

    def sum(self):
        """Computes the sum of the values of a tensor"""
        shape = self.shape
        if shape == (0,):
            raise Exception("Cannot compute the sum of the tensor since the tensor is empty")
        else:
            res = 0
            for index in loop_on_indicies(shape):
                res += self[index]
            return res

    def norm(self):
        """Computes the norm of a tensor, and also works if the tensor is multi-dimensional (then it acts as if the tensor were one-dimensional"""
        tensor = self**2
        return tensor.sum()**(1/2)

