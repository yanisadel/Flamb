import flamb
from .utils import *
import numpy as np

class Tensor(np.ndarray):
    # For the moment, Tensor can only contain flamb.Variable values

    def sum(self):
        shape = self.shape
        if shape == (0,):
            raise Exception("Cannot compute the sum of the tensor since the tensor is empty")
        else:
            res = 0
            for index in loop_on_indicies(shape):
                res += self[index]
            return res
