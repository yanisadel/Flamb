import numpy as np
import flamb
import random

def zeros(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in _loop_on_indicies(shape):
        tensor[index] = flamb.Variable(0, requires_grad=requires_grad)
    return tensor

def ones(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in _loop_on_indicies(shape):
        tensor[index] = flamb.Variable(1, requires_grad=requires_grad)
    return tensor

def rand(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in _loop_on_indicies(shape):
        tensor[index] = flamb.Variable(random.uniform(-1, 1), requires_grad=requires_grad)
    return tensor

def to_tensor(l, dtype=object):
    l = np.array(l, dtype=dtype)
    shape = l.shape
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in _loop_on_indicies(shape):
        tensor[index] = l[index]
    return tensor

def matmul(a, b):
    return np.matmul(a, b)
    
def dot(a, b):
    return np.dot(a, b)

def concatenate(a, b, axis=None):
    return np.concatenate((a, b), axis=axis)
    
def _loop_on_indicies(shape):
    nb_dim = len(shape)
    size = 1
    for elt in shape:
        size *= elt
    index = [0] * nb_dim
    # Indicateur pour savoir quand arrêter de parcourir l'array
    done = False
    if size == 0:
        done = True
    # Tant que nous n'avons pas fini de parcourir l'array
    while not done:
        yield tuple(index)
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