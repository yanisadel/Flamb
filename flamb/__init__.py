from flamb.autograd import Variable, no_grad
from flamb.tensor import *
from flamb import functional
from flamb import nn

environ = {"is_grad_enabled": True}

__all__ = [
    "Variable",
    "Tensor",
    "zeros",
    "ones",
    "no_grad",
    "environ",
    "functional",
    "nn",
]
