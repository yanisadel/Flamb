from copy import deepcopy


class Tensor:
    """
    A tensor is described by a shape (self.shape), and a list (self.data)
    self.nb_dim indicates the number of dimensions
    - self.size gives the total size of the tensor (corresponds to the product of the variables in self.shape)
    - self.respective_sizes give the slices for each dimension. 
    For instance, if a tensor is (2, 4, 4), if I want tensor[0], the indices of self.data I need are 0 and 16, so the respective size of the first dimension is 16
    """

    def __init__(self, l, shape=None):
        if shape is None:
            self.shape = Tensor._get_shape(l)
        else:
            self.shape = shape
        self.nb_dim = len(self.shape)
        self.size = Tensor._get_size(self.shape)
        self.respective_sizes = Tensor._get_respective_sizes(self.shape, self.size)

        if isinstance(l[0], list):
            self.data = Tensor._flatten(l)
        else:
            self.data = l

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
    def _get_size(shape):
        size = 1
        for dim in shape:
            size *= dim
        return size

    @staticmethod
    def _get_respective_sizes(shape, size):
        respective_sizes = []
        current_size = size

        for dim in shape:
            current_size = current_size // dim
            respective_sizes.append(current_size)

        return respective_sizes

    @staticmethod
    def _flatten(l):
        if isinstance(l, list):
            flatten_data = []
            for item in l:
                flatten_data.extend(Tensor._flatten(item))
            return flatten_data
        else:
            return [l]

    @staticmethod
    def _loop_on_indicies(shape):
        nb_dim = len(shape)
        index = [0] * nb_dim
        # Indicateur pour savoir quand arrêter de parcourir l'array
        done = False
        # Tant que nous n'avons pas fini de parcourir l'array
        while not done: 
            yield index
            index[-1] += 1
            for i in reversed(range(nb_dim)):
                if index[i] >= shape[i]:
                    index[i] = 0
                    # Si nous sommes à la dernière dimension, nous avons fini de parcourir l'array
                    if i == 0:
                        done = True

                    # Sinon, nous passons à la dimension suivante
                    else:
                        index[i - 1] += 1

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index,)

        if isinstance(index, (tuple, list)):
            nb_dim_asked = len(index)
            assert (
                nb_dim_asked <= self.nb_dim
            ), f"Too many dimensions (dimension of the array is {self.nb_dim}, and you asked for {nb_dim_asked} dimensions)"
            current_size = self.size
            pos = 0
            for shape_elt, index_elt in zip(self.shape, index):
                current_size = current_size // shape_elt
                pos += current_size * index_elt

            return Tensor(
                self.data[pos : pos + current_size], shape=self.shape[nb_dim_asked:]
            )

    def __setitem__(self, index, value):
        if isinstance(index, int):
            index = (index,)

        if isinstance(index, (tuple, list)):
            nb_dim_asked = len(index)
            assert (
                nb_dim_asked <= self.nb_dim
            ), f"Too many dimensions (dimension of the array is {self.nb_dim}, and you asked for {nb_dim_asked} dimensions)"

            if nb_dim_asked == self.nb_dim:
                # Test pour vérifier que c'est un entier
                pass
            elif nb_dim_asked < self.nb_dim:
                # Test pour vérifier que c'est un tenseur
                pass

            current_size = self.size
            pos = 0
            for shape_elt, index_elt in zip(self.shape, index):
                current_size = current_size // shape_elt
                pos += current_size * index_elt

            if nb_dim_asked:
                self.data[pos] = value
            else:
                self.data[pos : pos + current_size] = value

    def __eq__(self, value):
        # Only works with value that are not real tensors (tensors of dimension 0)
        assert len(self.data) == 1, "Cannot compare tensor and value"
        return self.data[0] == value

    def __repr__(self):
        # If dimension is 0, we just return tensor(value)
        if self.nb_dim == 0:
            return f"tensor({self.data[0]})"

        else:
            s = "tensor("
            # We add the first parenthesis
            for _ in range(self.nb_dim):
                s = s + "["

            index = [0] * self.nb_dim
            # Indicateur pour savoir quand arrêter de parcourir l'array
            done = False
            # Tant que nous n'avons pas fini de parcourir l'array
            while not done:
                # Afficher l'élément courant
                somme = 0
                for siz, ind in zip(self.respective_sizes, index):
                    somme += siz * ind

                s += f"{self.data[somme]}"

                index[-1] += 1
                nb_to_add = 0
                for i in reversed(range(self.nb_dim)):
                    if index[i] >= self.shape[i]:
                        s += "]"

                        index[i] = 0
                        # Si nous sommes à la dernière dimension, nous avons fini de parcourir l'array
                        if i == 0:
                            done = True

                        # Sinon, nous passons à la dimension suivante
                        else:
                            nb_to_add += 1
                            index[i - 1] += 1

                if not done:
                    s += ", "
                    for _ in range(nb_to_add):
                        s += "["

            s += ")"
            return s
