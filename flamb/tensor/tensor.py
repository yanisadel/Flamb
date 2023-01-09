import flamb
import numpy as np
from copy import deepcopy


class Tensor:
    """
    Does not arrays with uncorrect shape yet
    """

    def __init__(self, l, dtype=None, requires_grad=False):
        self.l = Tensor._initialize_tensor_variables(
            l, requires_grad=requires_grad, dtype=dtype
        )
        self.dtype = dtype
        self.shape = self.l.shape

    @staticmethod
    def _browse_tensor_indicies(shape):
        index = [0] * len(shape)
        # Indicateur pour savoir quand arrêter de parcourir l'array
        done = False
        # Tant que nous n'avons pas fini de parcourir l'array
        while not done:
            # Afficher l'élément courant
            yield tuple(index)

            # Passer à l'élément suivant
            index[0] += 1
            for i in range(len(shape)):
                if index[i] >= shape[i]:
                    index[i] = 0
                    # Si nous sommes à la dernière dimension, nous avons fini de parcourir l'array
                    if i == len(shape) - 1:
                        done = True
                    # Sinon, nous passons à la dimension suivante
                    else:
                        index[i + 1] += 1

    @staticmethod
    def _initialize_tensor_variables(l, requires_grad=False, dtype=None):
        # Récupérer la shape de l'array et initialiser la liste des indices à 0
        l = np.array(l)
        shape = l.shape
        tensor = np.empty(shape, dtype=object)

        for index in Tensor._browse_tensor_indicies(shape):
            # Afficher l'élément courant
            tensor[tuple(index)] = flamb.Variable(
                l[tuple(index)], dtype=dtype, requires_grad=requires_grad
            )

        return tensor

    def __getitem__(self, index):
        return self.l[index]

    def __setitem__(self, index, value):
        self.l[index] = value

    def __add__(self, new_variable):
        return Tensor(self.l + new_variable)

    def __radd__(self, new_variable):
        return self + new_variable

    def __iadd__(self, new_variable):
        return self + new_variable

    def __sub__(self, new_variable):
        return Tensor(self.l - new_variable)

    def __rsub__(self, new_variable):
        return Tensor(new_variable - self.l)

    def __isub__(self, new_variable):
        return self - new_variable

    def __mul__(self, new_variable):
        return Tensor(self.l * new_variable)

    def __rmul__(self, new_variable):
        return self * new_variable

    def __imul__(self, new_variable):
        return self * new_variable

    def __truediv__(self, new_variable):
        return Tensor(self.l / new_variable)

    def __rtruediv__(self, new_variable):
        return Tensor(new_variable / self.l)

    def __itruediv__(self, new_variable):
        return self / new_variable

    def sum(self):
        return self.l.sum()

    def reshape(self, shape):
        self.l.reshape(shape)

    def __repr__(self):
        return f"{self.l}"

