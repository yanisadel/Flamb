from flamb.autograd import Variable, no_grad, functional
from flamb.tensor import Tensor

environ = {"grad_enabled": True}

__all__ = ["Variable", "Tensor", "no_grad", "environ", "functional"]
