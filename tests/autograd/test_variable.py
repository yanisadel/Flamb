import flamb
from flamb import Variable
from flamb import functional as F
import math


def test_equality_operators():
    """Test of the operators =, <, >, >= and <="""
    x = Variable(5)
    assert x == 5
    assert x > 4
    assert x < 5.5
    assert x >= 4.2
    assert x <= 5


def test_operators():
    """Test of the operators +, +=, -, -=, *, *=, /, /="""
    value = 5.5
    x = Variable(value, requires_grad=False)
    assert x ** 2 == value ** 2, f"Value is not correct for {value}**2"
    assert (3 * x - 1) * (-1 * x + 2), f"Value is not correct for (3x - 1)(-x + 2)"
    assert (2 * x) / (x - 1), f"Value is not correct for (2x)/(x-1)"

    x = Variable(value)
    x += 3
    assert x == value + 3, "__iadd__ does not work for integers"

    x = Variable(value)
    x += x
    assert x == 2 * value, "__iadd__ does not work for variables"

    x = Variable(value)
    x -= 3
    assert x == value - 3, "__isub__ does not work"

    x = Variable(value)
    x *= 3
    assert x == value * 3, "__imul__ does not work"

    x = Variable(value)
    x /= 3
    assert x == value / 3, "__itruediv__ does not work"

    x = Variable(value)
    assert F.exp(x) == math.exp(value), f"Exp value is not correct for {value}"
    assert F.cos(x) == math.cos(value), f"Cos value is not correct for {value}"
    assert F.sin(x) == math.sin(value), f"Sin value is not correct for {value}"
    assert F.tan(x) == math.tan(value), f"Tan value is not correct for {value}"
    assert F.tanh(x) == math.tanh(value), f"Tanh value is not correct for {value}"


def test_elementary_gradients():
    x = Variable(4)
    y = 2 * x
    y.backward()
    assert x.grad == 2, "Gradient should be 2"

    x = Variable(5)
    y = x ** 3
    y.backward()
    assert x.grad == 75, "Gradient should be 75"

    x = Variable(2)
    y = 1 / x
    y.backward()
    assert x.grad == -1 / 4, "Gradient should be -1/4"


def test_math_functions_gradients():
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


    x = Variable(-2, requires_grad=True)
    y = F.ReLU(x)
    y.backward()
    assert x.grad == (
        0
    ), "Gradient of ReLU(-2) should be 0"
    x = Variable(5, requires_grad=True)
    y = F.ReLU(x)
    y.backward()
    assert x.grad == (
        1
    ), "Gradient of ReLU(5) should be 1"


def test_difficult_gradients():
    x = Variable(2)
    y = (x ** 2 - 1) ** 2
    y.backward()
    assert x.grad == 24, "Gradient should be 24"

    x = Variable(2)
    y = Variable(5)
    z = y / x ** 2
    z.backward()
    assert x.grad == -10 / 8, "Gradient should be -10/8"

    x = Variable(2)
    y = Variable(5)
    z = 1 / x + (x ** 2 - 1) ** 2 + y / x ** 2
    z.backward()
    assert x.grad == 22.5, "Gradient should be 22.5"


    value = 5
    x = Variable(value, requires_grad=True)
    y = (x ** 2) * F.cos(x)
    y.backward()
    assert x.grad == (
        (2 * value * F.cos(value) - (value ** 2) * F.sin(value))
    ), "Gradient should be (2x*cos(x) - x**2 * sin(x))"

    value = 5
    x = Variable(value, requires_grad=True)
    y = F.exp(x ** 2) * F.tanh(x)
    y.backward()
    assert abs(
        x.grad
        - (
            2 * value ** 1 * F.exp(value ** 2) * F.tanh(value)
            + F.exp(value ** 2) * (1 - F.tanh(value) ** 2)
        )
        < 1e-3
    ), "Gradient should be 3 * x**2 * exp(x**3) * tanh(x) + exp(x**3)*(1 - tanh(x)**2)"
    # < 1e-3 is because of the approximation errors

    value = 5
    x = Variable(value, requires_grad=True)
    y = (x ** 2) * F.cos(x) - F.exp(x ** 2) * F.tanh(x)
    y.backward()
    assert abs(
        x.grad
        - (
            (2 * value * F.cos(value) - (value ** 2) * F.sin(value))
            - (
                2 * value ** 1 * F.exp(value ** 2) * F.tanh(value)
                + F.exp(value ** 2) * (1 - F.tanh(value) ** 2)
            )
        )
        < 1e-3
    ), "Gradient should be ( (2x*cos(x) - x**2 * sin(x)) - ( 3 * x**2 * exp(x**3) * tanh(x) + exp(x**3)*(1 - tanh(x)**2) ) )"
    # < 1e-3 is because of the approximation errors


def test_reset_state():
    """Test that the reset_state method works"""
    x = Variable(5, requires_grad=True)
    y = (x*2 + 1)*(x - 1)
    y.reset_state()
    assert y.grad == 0
    assert y.last_operation is None


def test_inplace():
    """
    Test that if a variable is modified inplace (with a += for instance),
    and if we are in a flamb.no_grad context, then the variable is not erased (ie the address of the variables remains the same)
    
    Maybe in the future it would be better than the variable is not erased even if we are not in a no_grad context, but it makes gradient computation more difficult
    """
    x = Variable(5, requires_grad=True)
    first_id = id(x)
    y = x**2
    y.backward()
    with flamb.no_grad():
        x -= 1e-1*x.grad
        x.reset_state()
        x.requires_grad = True

    current_id = id(x)
    assert first_id == current_id


if __name__ == "__main__":
    test_equality_operators()
    test_operators()
    test_elementary_gradients()
    test_math_functions_gradients()
    test_difficult_gradients()
    test_reset_state()
    test_inplace()
