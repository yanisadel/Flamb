import flamb
from flamb import Tensor
from flamb.nn import losses

def test_value():
    loss = losses.MSE()
    a = flamb.zeros((3,3))
    b = flamb.ones((3,3))
    assert loss(a, a) == 0, f"Loss should be equal to 0 but is equal to {loss(a,a)}"
    assert loss(a, b) == 1, f"Loss should be equal to 1 but is equal to {loss(a,b)}"
    assert loss(a, 2*b) == 2, f"Loss should be equal to 2 but is equal to {loss(a,2*b)}"
    assert loss(a, 3*b) == 3, f"Loss should be equal to 2 but is equal to {loss(a,3*b)}"

if __name__ == '__main__':
    test_value()