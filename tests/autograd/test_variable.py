import flamb
from flamb import Variable
from flamb import functional as F
import math


def test_values():
    value = 5.5
    x = Variable(value, requires_grad=False)
    assert F.exp(x) == math.exp(value), f"Exp value is not correct for {value}"
    assert F.cos(x) == math.cos(value), f"Cos value is not correct for {value}"
    assert F.sin(x) == math.sin(value), f"Sin value is not correct for {value}"
    assert F.tan(x) == math.tan(value), f"Tan value is not correct for {value}"
    assert F.tanh(x) == math.tanh(value), f"Tanh value is not correct for {value}"


def test_gradients():
    x = Variable(4)
    y = 2 * x
    y.backward()
    assert x.grad == 2, "Gradient should be 2"

    x = Variable(5)
    y = x ** 3
    y.backward()
    assert x.grad == 75, "Gradient should be 75"

    x = Variable(2)
    y = Variable(5)
    z = 1 / x + (x ** 2 - 1) ** 2 + y / x ** 2
    z.backward()
    assert x.grad == 22.5, "Gradient should be 22.5"

    x = Variable(5, requires_grad=True)
    y = F.exp(x)
    y.backward()
    assert x.grad == math.exp(5), "Gradient of exp(5) should be exp(5)"

    x = Variable(5, requires_grad=True)
    y = F.cos(x)
    y.backward()
    assert x.grad == -math.sin(5), "Gradient of cos(5) should be -sin(5)"

    x = Variable(5, requires_grad=True)
    y = F.sin(x)
    y.backward()
    assert x.grad == math.cos(5), "Gradient of sin(5) should be cos(5)"

    x = Variable(5, requires_grad=True)
    y = F.tan(x)
    y.backward()
    assert y == math.tan(5)
    assert x.grad == (
        1 + math.tan(5) ** 2
    ), "Gradient of tan(5) should be 1 + tan(5)**2"

    x = Variable(5, requires_grad=True)
    y = F.tanh(x)
    y.backward()
    assert x.grad == (
        1 - math.tanh(5) ** 2
    ), "Gradient of tanh(5) should be 1 - tanh(5)**2"


if __name__ == "__main__":
    test_values()
    test_gradients()
