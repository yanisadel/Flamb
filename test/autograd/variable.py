import flamb
from flamb import Variable

def test_variable_gradient():
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

if __name__ == '__main__':
    test_variable_gradient()
