import flamb
from copy import deepcopy

class Tensor:
    """
    Does not arrays with uncorrect shape yet
    """
    def __init__(self, l, dtype=None, requires_grad=False):
        self.l = deepcopy(list(l))
        self.dtype = dtype
        self.shape = Tensor._get_shape(self.l)
        Tensor._initialize_tensor_variables(self.l, requires_grad=requires_grad, dtype=dtype)
        

    @staticmethod
    def _get_shape(l):
        shape = []
        done = False
        current_list = deepcopy(l)
        while not done:
            try:
                shape.append(len(current_list))
                current_list = current_list[0]
            except:
                done = True
        
        return tuple(shape)

    @staticmethod
    def _initialize_tensor_variables(l, requires_grad=False, dtype=None):
        n = len(l)
        if isinstance(l[0], list):
            for i in range(n):
                Tensor._initialize_tensor_variables(l[i], requires_grad=requires_grad, dtype=dtype)

        else:
            for i in range(n):
                l[i] = flamb.Variable(l[i], dtype=dtype, requires_grad=requires_grad)


    def __repr__(self):
        return f"{self.l}"

