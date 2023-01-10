from flamb.autograd import Variable, no_grad
from flamb.tensor import Tensor
from flamb import functional

environ = {"is_grad_enabled": True}

__all__ = ["Variable", "Tensor", "no_grad", "environ", "functional"]
