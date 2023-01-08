from flamb import Variable

class Tensor:
    def __init__(self, l, dtype=None, requires_grad=False):
        self.l = Tensor.fill(l, requires_grad=requires_grad)
        self.dtype = dtype

    @staticmethod
    def fill(l, requires_grad=False):
        n = len(l)
        for i in range(n):
            if isinstance(l[i], list):
                l[i] = Tensor.fill(l[i])
            else:
                l[i] = Variable(l[i], requires_grad=requires_grad)

        return l


    def shape(self):
        pass

    def __repr__(self):
        return f"{self.l}"
            