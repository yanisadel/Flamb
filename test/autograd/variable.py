import flamb
from flamb import Variable
from flamb import functional as F
import math 

def test_gradient_calculation():
    x = Variable(4)
    y = 2*x
    y.backward()
    assert (x.grad == 2)

    x = Variable(5)
    y = x**3
    y.backward()
    assert (x.grad == 75)

    x = Variable(2)
    y = Variable(5)
    z = 1/x + (x**2 - 1)**2 + y/x**2
    z.backward()
    assert x.grad == 22.5


def test_exp():
    x = Variable(5, requires_grad=True)
    y = F.exp(x)
    y.backward()
    assert y == math.exp(5)
    assert x.grad == math.exp(5)

def test_cos():
    x = Variable(5, requires_grad=True)
    y = F.cos(x)
    y.backward()
    assert y == math.cos(5)
    assert x.grad == -math.sin(5)

def test_sin():
    x = Variable(5, requires_grad=True)
    y = F.sin(x)
    y.backward()
    assert y == math.sin(5)
    assert x.grad == math.cos(5)

def test_tanh():
    x = Variable(5, requires_grad=True)
    y = F.tanh(x)
    y.backward()
    assert y == math.tanh(5)
    assert x.grad == (1 - math.tanh(5)**2)


if __name__ == '__main__':
    test_gradient_calculation()
    test_exp()
    test_cos()
    test_sin()
    test_tanh()
