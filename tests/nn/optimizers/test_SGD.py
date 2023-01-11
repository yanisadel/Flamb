import flamb
from flamb import Variable, Tensor
from flamb.nn.optimizers import SGD


def test_value():
    """
    Test that the value of variables modified by SGD is correct,
    and that the address of the variables are still the same
    """
    x = Variable(4)
    y = Variable(2)
    first_id = id(x)
    parameters = flamb.to_tensor([x, y])
    optimizer = SGD(parameters, learning_rate=1e-1)
    assert optimizer.learning_rate == 1e-1

    loss = x**2 + y**2
    loss.backward()
    optimizer.step()
    assert x == 3.2
    assert y == 1.6

    current_id = id(x)
    assert first_id == current_id

if __name__ == '__main__':
    test_value()


