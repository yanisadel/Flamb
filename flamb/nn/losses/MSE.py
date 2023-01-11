from .base import Loss

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        assert x.shape == y.shape, "Shape of x and y are not the same"
        diff = (x - y)**2
        sum = diff.sum()
        return sum**(1/2)