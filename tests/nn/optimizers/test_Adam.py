import flamb
from flamb import Variable, Tensor
from flamb.nn.optimizers import Adam


def test_value():
    """
    Test that the value of variables modified by Adam is correct,
    and that the address of the variables are still the same
    """
    x = Variable(4)
    y = Variable(2)
    first_id = id(x)
    parameters = flamb.to_tensor([x, y])
    learning_rate = 1e-1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-7
    optimizer = Adam(parameters, learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)
    assert ((optimizer.learning_rate == learning_rate) and (optimizer.beta1 == beta1) and (optimizer.beta2 == beta2))

    loss = x**2 + y**2
    loss.backward()
    optimizer.step()
    second_momentum_norm_sq = 8**2 + 4**2
    new_x = 4 - learning_rate * (1 - beta1)*2*4 / ( ( (1 - beta2) * second_momentum_norm_sq ) ** (1/2) + eps)
    new_y = 2 - learning_rate * (1 - beta1)*2*2 / ( ( (1 - beta2) * second_momentum_norm_sq ) ** (1/2) + eps)
    assert abs(x.value - new_x) < 1e-4 # < 1e-4 for approximation errors
    assert abs(y.value - new_y) < 1e-4

    assert id(x) == first_id


if __name__ == '__main__':
    test_value()



