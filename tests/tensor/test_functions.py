import flamb

def test_zeros():
    """Test the function that generates a tensor of zeros"""
    shape = (2, 2)
    tensor = flamb.zeros(shape)
    assert tensor.shape == shape, "Shape is not correct"
    for i in range(2):
        for j in range(2):
            assert tensor[i][j] == 0


def test_ones():
    """Test the function that generates a tensor of ones"""
    shape = (2, 2)
    tensor = flamb.ones(shape)
    assert tensor.shape == shape, "Shape is not correct"
    for i in range(2):
        for j in range(2):
            assert tensor[i][j] == 1


def test_to_tensor():
    """Test the function that converts an array-like to a tensor"""
    l = [ [1,2],
          [3,4]]
    tensor = flamb.to_tensor(l)

    assert isinstance(tensor, flamb.Tensor)
    assert tensor.shape == (2,2)
    for i in range(2):
        for j in range(2):
            assert tensor[i][j] == l[i][j]


def test_concatenate():
    """Test the function that concatenates two tensors"""
    a = flamb.to_tensor([1,2])
    b = flamb.to_tensor([3,4])
    res = flamb.concatenate(a, b)
    target = flamb.to_tensor([1,2,3,4])

    assert res.shape == (4,)
    for i in range(4):
        assert res[i] == target[i]
    

if __name__ == "__main__":
    test_zeros()
    test_ones()
    test_to_tensor()
    test_concatenate()
