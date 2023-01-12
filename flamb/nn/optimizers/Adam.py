import flamb
from .base import Optimizer

class Adam(Optimizer):
    """
    Adam algorithm
    """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-7):
        self.params = params
        self.nb_params = len(self.params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_momentum = flamb.zeros((self.nb_params,))
        self.second_momentum = 0

    def step(self):
        with flamb.no_grad():
            grads = flamb.to_tensor([param.grad for param in self.params])
            self.first_momentum = self.beta1 * self.first_momentum + (1 - self.beta1) * grads
            self.second_momentum = self.beta2 * self.second_momentum + (1 - self.beta2) * (grads.norm()**2)
            for i in range(self.nb_params):
                self.params[i] -= self.learning_rate * self.first_momentum[i] / (self.second_momentum**(1/2) + self.eps)
                self.params[i].reset_state(requires_grad=True)