class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise Exception("You need to implement the step method")