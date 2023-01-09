import flamb
from flamb import Tensor
from copy import deepcopy


def test_shape():
    l = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    l = Tensor(l, requires_grad=True)

    assert l.shape == (4, 3)


if __name__ == "__main__":
    test_shape()
