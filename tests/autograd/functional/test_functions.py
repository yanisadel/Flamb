import flamb
from flamb import Variable
from flamb import functional as F

import math


def test_exp():
    x = Variable(5)
    y = 5
    assert F.exp(x) == math.exp(5)
    assert F.exp(y) == math.exp(y)


def test_cos():
    x = Variable(5)
    y = 5
    assert F.cos(x) == math.cos(5)
    assert F.cos(y) == math.cos(y)


def test_sin():
    x = Variable(5)
    y = 5
    assert F.sin(x) == math.sin(5)
    assert F.sin(y) == math.sin(y)


def test_tan():
    x = Variable(5)
    y = 5
    assert F.tan(x) == math.tan(5)
    assert F.tan(y) == math.tan(y)


def test_tanh():
    x = Variable(5)
    y = 5
    assert F.tanh(x) == math.tanh(5)
    assert F.tanh(y) == math.tanh(y)


if __name__ == "__main__":
    test_exp()
    test_cos()
    test_sin()
    test_tan()
    test_tanh()
