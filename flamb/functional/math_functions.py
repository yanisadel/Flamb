"""
This file contains mathematical functions that work on Variable, or classical types like int and float
"""

import flamb
import math
import numpy as np

def exp(x):
    if isinstance(x, flamb.Tensor):
        return np.vectorize(math.exp)(x)
    elif isinstance(x, flamb.Variable):
        return x.exp()
    else:
        return math.exp(x)


def cos(x):
    if isinstance(x, flamb.Tensor):
        return np.vectorize(math.cos)(x)
    elif isinstance(x, flamb.Variable):
        return x.cos()
    else:
        return math.cos(x)


def sin(x):
    if isinstance(x, flamb.Tensor):
        return np.vectorize(math.sin)(x)
    elif isinstance(x, flamb.Variable):
        return x.sin()
    else:
        return math.sin(x)


def tan(x):
    if isinstance(x, flamb.Tensor):
        return np.vectorize(math.tan)(x)
    elif isinstance(x, flamb.Variable):
        return x.tan()
    else:
        return math.tan(x)


def tanh(x):
    if isinstance(x, flamb.Tensor):
        return np.vectorize(math.tanh)(x)
    elif isinstance(x, flamb.Variable):
        return x.tanh()
    else:
        return math.tanh(x)