import flamb
from flamb import nn


def test_shape():
    x = flamb.zeros(shape=(8, 20))
    layer = nn.Linear(20, 30)
    output = layer(x)

    assert (output.shape == (8, 30)), f"Shape should be (8, 30) but is {output.shape}"

def test_values():
    x = flamb.to_tensor([2,2])
    layer = nn.Linear(2, 2)
    layer.weights[0][0] = 10
    layer.weights[0][1] = 100
    layer.weights[1][0] = 1000
    layer.weights[1][1] = 10000
    layer.bias[0] = 0
    layer.bias[1] = 0
    output = layer(x)
    assert output[0] == 2020 and output[1] == 20200, "Error on dot product"

    layer.bias[0] = -5
    layer.bias[1] = -10
    output = layer(x)
    assert output[0] == 2015 and output[1] == 20190, "Error on dot product"

def test_get_parameters():
    layer = nn.Linear(2, 2)
    layer.weights[0][0] = 10
    layer.weights[0][1] = 100
    layer.weights[1][0] = 1000
    layer.weights[1][1] = 10000
    layer.bias[0] = 0
    layer.bias[1] = 0

    parameters = layer.get_parameters()
    assert parameters.shape == (6,)
    assert parameters[0] == 10
    assert parameters[1] == 100
    assert parameters[2] == 1000
    assert parameters[3] == 10000
    assert parameters[4] == 0
    assert parameters[5] == 0


if __name__ == '__main__':
    test_shape()
    test_values()
    test_get_parameters()