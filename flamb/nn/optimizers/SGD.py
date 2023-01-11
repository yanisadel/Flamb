import flamb
from .base import Optimizer

class SGD(Optimizer):
    """
    Performs the Stochastic Gradient Descent algorithm
    """
    def __init__(self, params, learning_rate=1e-3):
        self.params = params
        self.nb_params = len(self.params)
        self.learning_rate = learning_rate

    def step(self):
        with flamb.no_grad():
            for i in range(self.nb_params):
                self.params[i] -= self.learning_rate*self.params[i].grad
                self.params[i].requires_grad = True
                self.params[i].grad = 0
                self.params[i].last_operation = None
                
