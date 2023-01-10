import flamb


def zeros(shape, requires_grad=False):
    size = 1
    for elt in shape:
        size *= elt

    tensor = flamb.Tensor(
        [flamb.Variable(0, requires_grad=requires_grad) for _ in range(size)],
        shape=shape,
    )
    return tensor


def ones(shape, requires_grad=False):
    size = 1
    for elt in shape:
        size *= elt

    tensor = flamb.Tensor(
        [flamb.Variable(1, requires_grad=requires_grad) for _ in range(size)],
        shape=shape,
    )
    return tensor
