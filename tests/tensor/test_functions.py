import flamb


def test_zeros():
    shape = (2, 2)
    tensor = flamb.zeros(shape)
    assert tensor.shape == shape, "Shape is not correct"
    for i in range(2):
        for j in range(2):
            assert tensor[i][j] == 0


def test_ones():
    shape = (2, 2)
    tensor = flamb.ones(shape)
    assert tensor.shape == shape, "Shape is not correct"
    for i in range(2):
        for j in range(2):
            assert tensor[i][j] == 1


if __name__ == "__main__":
    test_zeros()
    test_ones()
