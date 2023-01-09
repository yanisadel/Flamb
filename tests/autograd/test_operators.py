from flamb.autograd.operators import *
from flamb import Variable

import math


def test_sum():
    x, y, z = 2, 3, 4
    operator = SumOperator(x, y, z)
    assert operator.variables == [x, y, z], "Variables are not correct"
    assert operator.gradient() == [1, 1, 1], "Gradient is not correct"

    x, y, z = Variable(2), Variable(3), Variable(4)
    operator = SumOperator(x, y, z)
    assert operator.variables == [x, y, z], "Variables are not correct"
    assert operator.gradient() == [1, 1, 1], "Gradient is not correct"


def test_product():
    x, y, z = 2, 3, 4
    operator = ProductOperator(x, y, z)
    assert operator.variables == [x, y, z], "Variables are not correct"
    assert operator.gradient() == [y * z, x * z, x * y], "Gradient is not correct"

    x, y, z = Variable(2), Variable(3), Variable(4)
    operator = ProductOperator(x, y, z)
    assert operator.variables == [x, y, z], "Variables are not correct"
    assert operator.gradient() == [y * z, x * z, x * y], "Gradient is not correct"


def test_division():
    x, y = Variable(3), Variable(4)
    operator = DivisionOperator(x, y)
    assert operator.variables == [x, y], "Variables are not correct"
    assert operator.gradient() == [1 / y, x * (-1) / y ** 2], "Gradient is not correct"


def test_power():
    x, power = Variable(2), 3
    operator = PowerOperator(x, power)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [power * x ** (power - 1)], "Gradient is not correct"


def test_exp():
    x = Variable(3)
    operator = ExpOperator(x)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [math.exp(3)], "Gradient is not correct"


def test_cos():
    x = Variable(3)
    operator = CosOperator(x)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [-math.sin(3)], "Gradient is not correct"


def test_sin():
    x = Variable(3)
    operator = SinOperator(x)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [math.cos(3)], "Gradient is not correct"


def test_tan():
    x = Variable(3)
    operator = TanOperator(x)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [1 + math.tan(3) ** 2], "Gradient is not correct"


def test_tanh():
    x = Variable(3)
    operator = TanhOperator(x)
    assert operator.variables == [x], "Variables are not correct"
    assert operator.gradient() == [1 - math.tanh(3) ** 2], "Gradient is not correct"


if __name__ == "__main__":
    test_sum()
    test_product()
    test_division()
    test_power()
    test_exp()
    test_cos()
    test_sin()
    test_tan()
    test_tanh()
