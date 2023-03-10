import numpy as np
import flamb
import random
from .utils import *

def zeros(shape, dtype=object, requires_grad=False):
    """Creates a tensor composed of zeros"""
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(0, requires_grad=requires_grad)
    return tensor


def ones(shape, dtype=object, requires_grad=False):
    """Creates a tensor composed of ones"""
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(1, requires_grad=requires_grad)
    return tensor


def rand(shape, dtype=object, requires_grad=False):
    """Creates a tensor composed of random values between -1 and 1"""
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(random.uniform(-1, 1), requires_grad=requires_grad)
    return tensor


def to_tensor(l, dtype=object, requires_grad=False):
    """Convert an array-like object (a list or a numpy array) to a tensor"""
    l = np.array(l, dtype=dtype)
    shape = l.shape
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        if not isinstance(l[index], flamb.Variable):
            tensor[index] = flamb.Variable(l[index], requires_grad=requires_grad)
        else:
            tensor[index] = l[index]
    return tensor


def matmul(a, b):
    """Computes the matrix multiplication of a and b"""
    return np.matmul(a, b)

    
def dot(a, b):
    """Computes the dot product of a and b"""
    return np.dot(a, b)


def concatenate(a, b, axis=None):
    """Concatenates the tensors a and b"""
    return np.concatenate((a, b), axis=axis)
    