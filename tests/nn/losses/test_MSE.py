import flamb
from flamb import Tensor
from flamb.nn import losses

def test_MSE():
    loss = losses.MSE()
    a = flamb.zeros((3,3))
    b = flamb.ones((3,3))
    assert loss(a, b) == 1, f"Loss should be equal to 1 but is equal to {loss(a,b)}"
    assert loss(a, 2*b) == 2, f"Loss should be equal to 2 but is equal to {loss(a,b)}"
    assert loss(a, 3*b) == 3, f"Loss should be equal to 2 but is equal to {loss(a,b)}"

if __name__ == '__main__':
    test_MSE()