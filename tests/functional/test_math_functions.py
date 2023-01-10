import flamb
from flamb import Variable, Tensor
from flamb import functional as F

import math
from copy import deepcopy

l_ref = [[1, 2, 3], [4, 5, 6]]


def test_exp():
    x = Variable(5)
    y = 5
    assert F.exp(x) == math.exp(5)
    assert F.exp(y) == math.exp(y)

    l = flamb.to_tensor(l_ref)
    l = F.exp(l)
    assert l[0][0] == math.exp(1)


def test_cos():
    x = Variable(5)
    y = 5
    assert F.cos(x) == math.cos(5)
    assert F.cos(y) == math.cos(y)

    l = flamb.to_tensor(l_ref)
    l = F.cos(l)
    assert l[0][0] == math.cos(1)


def test_sin():
    x = Variable(5)
    y = 5
    assert F.sin(x) == math.sin(5)
    assert F.sin(y) == math.sin(y)

    l = flamb.to_tensor(l_ref)
    l = F.sin(l)
    assert l[0][0] == math.sin(1)


def test_tan():
    x = Variable(5)
    y = 5
    assert F.tan(x) == math.tan(5)
    assert F.tan(y) == math.tan(y)

    l = flamb.to_tensor(l_ref)
    l = F.tan(l)
    assert l[0][0] == math.tan(1)


def test_tanh():
    x = Variable(5)
    y = 5
    assert F.tanh(x) == math.tanh(5)
    assert F.tanh(y) == math.tanh(y)

    l = flamb.to_tensor(l_ref)
    l = F.tanh(l)
    assert l[0][0] == math.tanh(1)


if __name__ == "__main__":
    test_exp()
    test_cos()
    test_sin()
    test_tan()
    test_tanh()
