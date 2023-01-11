import flamb
from flamb import Tensor
from flamb.nn import losses

def test_MSE():
    loss = losses.MSE()
    a = flamb.zeros((3,3))
    b = flamb.ones((3,3))
    assert loss(a, b) == 3, f"Loss should be equal to 3 but is equal to {loss(a,b)}"
    assert loss(a, 2*b) == 6, f"Loss should be equal to 6 but is equal to {loss(a,b)}"

if __name__ == '__main__':
    test_MSE()