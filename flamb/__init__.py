from flamb.autograd import Variable, no_grad
from flamb.tensor import *

environ = {'compute_grad': True}

__all__ = ['Variable', 'Tensor', 'no_grad', 'environ']
