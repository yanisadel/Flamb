import numpy as np
import flamb
import random
from .utils import *

def zeros(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(0, requires_grad=requires_grad)
    return tensor

def ones(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(1, requires_grad=requires_grad)
    return tensor

def rand(shape, dtype=object, requires_grad=False):
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(random.uniform(-1, 1), requires_grad=requires_grad)
    return tensor

def to_tensor(l, requires_grad=False, dtype=object):
    l = np.array(l, dtype=dtype)
    shape = l.shape
    tensor = flamb.Tensor(shape, dtype=dtype)
    for index in loop_on_indicies(shape):
        tensor[index] = flamb.Variable(l[index], requires_grad=requires_grad)
    return tensor

def matmul(a, b):
    return np.matmul(a, b)
    
def dot(a, b):
    return np.dot(a, b)

def concatenate(a, b, axis=None):
    return np.concatenate((a, b), axis=axis)
    