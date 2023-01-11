"""
This file defines a Variable class, which allows to track the gradient of a variable with respect to another one
"""

import flamb
from .operators import *
from flamb.utils import *
import math


class Variable:
    """
    A Variable is defined by
    - value (int or float)
    - dtype : the type we want for the value
    - requires_grad (bool) : True or False
    - last_operation (flamb.autograd.operators.BaseOperator) : None or an instance of BaseOperator
    """

    def __init__(self, value, dtype=None, requires_grad=True, last_operation=None):
        self.value = value
        if dtype:
            self.dtype = dtype
        else:
            self.dtype = type(self.value)
        self.requires_grad = requires_grad
        if not flamb.environ["is_grad_enabled"]:
            self.requires_grad = False

        self.grad = 0
        self.last_operation = last_operation

    def __repr__(self):
        return f"{self.value}"

    def __add__(self, var, inplace=False):
        """
        The inplace parameter is equal to False if we want a new variable to be created,
        or inplace=True if we want the value to be modified in the current instance.
        If inplace=True, then the gradient is set back to 0 and the last operations made on the instance are forgotten
        """
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(var, Variable):
            new_value += var.value
            requires_grad = requires_grad or var.requires_grad
        elif isinstance(var, (int, float)):
            new_value += var
        else:
            raise Exception(f"Cannot sum a {self.__class__} and a {type(var)}")

        last_operation = SumOperator(self, var)

        if inplace:
            self.value = new_value
            self.grad = 0
            self.requires_grad = False
            self.last_operation = None
            return self

        else:
            return Variable(
                new_value, requires_grad=requires_grad, last_operation=last_operation,
            )

    def __radd__(self, var):
        return self + var

    def __iadd__(self, var):
        if flamb.environ['is_grad_enabled']:
            return self.__add__(var, inplace=False)
        else:
            return self.__add__(var, inplace=True)

    def __sub__(self, var, inplace=False):
        negative_var = var * (-1)
        return self.__add__(negative_var, inplace=inplace)

    def __rsub__(self, var):
        negative_value = self * (-1)
        return negative_value + var

    def __isub__(self, var):
        if flamb.environ['is_grad_enabled']:
            return self.__sub__(var, inplace=False)
        else:
            return self.__sub__(var, inplace=True)

    def __mul__(self, var, inplace=False):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(var, Variable):
            new_value *= var.value
            requires_grad = requires_grad or var.requires_grad
        elif isinstance(var, (int, float)):
            new_value *= var
        else:
            raise Exception(f"Cannot multiply a {self.__class__} and a {type(var)}")

        last_operation = ProductOperator(self, var)
        if inplace:
            self.value = new_value
            self.grad = 0
            self.requires_grad = False
            self.last_operation = None
            return self

        else:
            return Variable(
                new_value, requires_grad=requires_grad, last_operation=last_operation,
            )

    def __rmul__(self, var):
        return self * var

    def __imul__(self, var):
        if flamb.environ['is_grad_enabled']:
            return self.__mul__(var, inplace=False)
        else:
            return self.__mul__(var, inplace=True)

    def __truediv__(self, var, inplace=False):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(var, Variable):
            new_value /= var.value
            requires_grad = requires_grad or var.requires_grad
        elif isinstance(var, (int, float)):
            new_value /= var
        else:
            raise Exception(f"Cannot divide a {self.__class__} by a {type(var)}")

        last_operation = DivisionOperator(self, var)

        if inplace:
            self.value = new_value
            self.grad = 0
            self.requires_grad = False
            self.last_operation = None
            return self

        else:
            return Variable(
                new_value, requires_grad=requires_grad, last_operation=last_operation,
            )

    def __rtruediv__(self, var, inplace=False):
        new_value = 0
        requires_grad = self.requires_grad
        if isinstance(var, Variable):
            new_value = var.value / self.value
            requires_grad = requires_grad or var.requires_grad
        elif isinstance(var, (int, float)):
            new_value = var / self.value
        else:
            raise Exception(f"Cannot divide a {type(var)} by a {self.__class__}")

        last_operation = DivisionOperator(var, self)
        if inplace:
            self.value = new_value
            self.grad = 0
            self.requires_grad = False
            self.last_operation = None
            return self

        else:
            return Variable(
                new_value, requires_grad=requires_grad, last_operation=last_operation,
            )

    def __itruediv__(self, var):
        if flamb.environ['is_grad_enabled']:
            return self.__truediv__(var, inplace=False)
        else:
            return self.__truediv__(var, inplace=True)

    def __floordiv__(self, var):
        raise Exception(r"The operation // is not implemented yet")

    def __pow__(self, power):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(power, Variable):
            new_value = new_value ** (power.value)
            requires_grad = requires_grad or power.requires_grad
        elif isinstance(power, (int, float)):
            new_value = new_value ** power
        else:
            raise Exception(
                f"Cannot calculate a {self.__class__} to the power of a {type(power)}"
            )

        last_operation = PowerOperator(self, power)
        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def __eq__(self, var):
        """="""
        var = convert_variable(var)
        return self.value == var

    def __ne__(self, var):
        """!="""
        var = convert_variable(var)
        return self.value != var

    def __gt__(self, var):
        """>"""
        var = convert_variable(var)
        return self.value > var

    def __ge__(self, var):
        """>="""
        var = convert_variable(var)
        return self.value >= var

    def __lt__(self, var):
        """<"""
        var = convert_variable(var)
        return self.value < var

    def __le__(self, var):
        """<="""
        var = convert_variable(var)
        return self.value <= var

    def exp(self):
        new_value = math.exp(self.value)
        requires_grad = self.requires_grad
        last_operation = ExpOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def cos(self):
        new_value = math.cos(self.value)
        requires_grad = self.requires_grad
        last_operation = CosOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def sin(self):
        new_value = math.sin(self.value)
        requires_grad = self.requires_grad
        last_operation = SinOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def tan(self):
        new_value = math.tan(self.value)
        requires_grad = self.requires_grad
        last_operation = TanOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def tanh(self):
        new_value = math.tanh(self.value)
        requires_grad = self.requires_grad
        last_operation = TanhOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def ReLU(self):
        new_value = max(self.value, 0)
        requires_grad = self.requires_grad
        last_operation = ReLUOperator(self)

        return Variable(
            new_value, requires_grad=requires_grad, last_operation=last_operation,
        )

    def backward(self, accumulated_grad=None):
        """
        The current variable was obtained with some variables.
        This function computes the gradient of the current variable with respect to the other variables

        Parameters
        ----------
            accumulated_grad (float) : default=None. The gradient of the current variable that has been computed.
                                       It allows to use chain rule.
        """
        if flamb.environ["is_grad_enabled"]:
            if accumulated_grad == None:
                accumulated_grad = 1

            self.grad += accumulated_grad

            last_operation = self.last_operation
            if last_operation != None:
                variables = last_operation.get_variables()
                grads = last_operation.gradient()

                for var, grad in zip(variables, grads):
                    if isinstance(var, Variable) and var.requires_grad:
                        var.backward(accumulated_grad * grad)

        else:
            raise Exception(
                "Cannot compute gradient in because grad is disabled (you're probably in a flamb.no_grad context)"
            )
    

    def reset_state(self, requires_grad=False):
        self.grad = 0
        self.last_operation = None
        self.requires_grad = requires_grad
