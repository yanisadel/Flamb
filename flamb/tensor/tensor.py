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
        Tensor._initialize_tensor_variables(self, requires_grad=requires_grad, dtype=dtype)
        
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

    """
    Version récursive
    @staticmethod
    def _initialize_tensor_variables(l, requires_grad=False, dtype=None):
        n = len(l)
        if isinstance(l[0], list):
            for i in range(n):
                Tensor._initialize_tensor_variables(l[i], requires_grad=requires_grad, dtype=dtype)

        else:
            for i in range(n):
                l[i] = flamb.Variable(l[i], dtype=dtype, requires_grad=requires_grad)
    """
    
    def _initialize_tensor_variables(self, requires_grad=False, dtype=None):
        # Récupérer la shape de l'array et initialiser la liste des indices à 0
        shape = self.shape
        index = [0] * len(shape)
        
        # Indicateur pour savoir quand arrêter de parcourir l'array
        done = False
        
        # Tant que nous n'avons pas fini de parcourir l'array
        while not done:
            # Afficher l'élément courant
            self[tuple(index)] = flamb.Variable(self[tuple(index)], dtype=dtype, requires_grad=requires_grad)

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


    def __getitem__(self, index):
        if isinstance(index, int):
            return self.l[index]
        elif isinstance(index, (tuple, list)):
            res = self.l
            for elt in index:
                res = res[elt]
            return res

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.l[index] = value
        elif isinstance(index, (tuple, list)):
            n = len(index)
            res = self.l
            for i in range(n-1):
                res = res[index[i]]
            res[index[-1]] = value
        else:
            raise Exception("The index must be an integer, a tuple or a list")


    def __repr__(self):
        return f"Tensor({self.l}, dtype={self.dtype})"

