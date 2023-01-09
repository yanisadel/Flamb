import flamb
from flamb import Tensor
from copy import deepcopy

def test_shape():
    l = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    l = Tensor(l, requires_grad=True)

    assert l.shape == (4,3)

if __name__ == '__main__':
    test_shape()




import numpy as np

def parcours_array(array):
    # Récupérer la shape de l'array et initialiser la liste des indices à 0
    shape = list(array.shape)
    indice = [0] * len(shape)
    
    # Indicateur pour savoir quand arrêter de parcourir l'array
    fini = False
    
    # Tant que nous n'avons pas fini de parcourir l'array
    while not fini:
        # Afficher l'élément courant
        print(array[tuple(indice)])

        # Passer à l'élément suivant
        indice[0] += 1
        for i in range(len(shape)):
            if indice[i] >= shape[i]:
                indice[i] = 0
                # Si nous sommes à la dernière dimension, nous avons fini de parcourir l'array
                if i == len(shape) - 1:
                    fini = True
                # Sinon, nous passons à la dimension suivante
                else:
                    indice[i + 1] += 1





l = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
parcours_array(l)