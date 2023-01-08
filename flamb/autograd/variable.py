import flamb
from flamb.autograd.operators import *
from flamb.utils import *

class Variable:
    def __init__(self, value, dtype=None, requires_grad=True, last_operation=None):
        self.value = value
        if dtype:
            self.dtype = dtype
        else:
            self.dtype = type(self.value)
        self.requires_grad = requires_grad
        if not flamb.environ['compute_grad']:
            self.requires_grad = False

        self.grad = 0
        self.last_operation = last_operation

    def __repr__(self):
        return f"Variable({self.value}, dtype={self.dtype.__name__}, requires_grad={self.requires_grad})"

    def __add__(self, new_variable):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(new_variable, Variable):
            new_value += new_variable.value
            requires_grad = requires_grad or new_variable.requires_grad
        else:
            new_value += new_variable
        
        dtype = type(new_value)

        if flamb.environ['compute_grad']:
            last_operation = SumOperator(self, new_variable)
            return Variable(new_value, dtype=dtype, requires_grad=requires_grad, last_operation=last_operation)
        else:
            self.__init__(new_value, dtype=dtype, requires_grad=False, last_operation=None)

    def __radd__(self, new_variable):
        return self + new_variable

    def __iadd__(self, new_variable):
        return self + new_variable



    def __sub__(self, new_variable):
        negative_value_to_add = new_variable*(-1)
        return self + negative_value_to_add

    def __rsub__(self, new_variable):
        negative_value = self*(-1)
        return negative_value + new_variable

    def __isub__(self, new_variable):
        return self - new_variable



    def __mul__(self, new_variable):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(new_variable, Variable):
            new_value *= new_variable.value
            requires_grad = requires_grad or new_variable.requires_grad
        else:
            new_value *= new_variable
        
        dtype = type(new_value)

        if flamb.environ['compute_grad']:
            last_operation = ProductOperator(self, new_variable)
            return Variable(new_value, dtype=dtype, requires_grad=requires_grad, last_operation=last_operation)
        else:
            self.__init__(new_value, dtype=dtype, requires_grad=False, last_operation=None)

    def __rmul__(self, new_variable):
        return self*new_variable

    def __imul__(self, new_variable):
        return self*new_variable



    def __truediv__(self, new_variable):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(new_variable, Variable):
            new_value /= new_variable.value
            requires_grad = requires_grad or new_variable.requires_grad
        else:
            new_value /= new_variable
        
        dtype = type(new_value)

        if flamb.environ['compute_grad']:
            last_operation = DivisionOperator(self, new_variable)
            return Variable(new_value, dtype=dtype, requires_grad=requires_grad, last_operation=last_operation)
        else:
            self.__init__(new_value, dtype=dtype, requires_grad=False, last_operation=None)

    def __rtruediv__(self, new_variable):
        new_value = 0
        requires_grad = self.requires_grad
        if isinstance(new_variable, Variable):
            new_value = new_variable.value / self.value
            requires_grad = requires_grad or new_variable.requires_grad
        else:
            new_value = new_variable / self.value
        
        dtype = type(new_value)

        if flamb.environ['compute_grad']:
            last_operation = DivisionOperator(new_variable, self)
            return Variable(new_value, dtype=dtype, requires_grad=requires_grad, last_operation=last_operation)
        else:
            self.__init__(new_value, dtype=dtype, requires_grad=False, last_operation=None)


    def __floordiv__(self, new_variable):
        raise Exception(r"The operation // is not implemented yet")


    def __pow__(self, power):
        new_value = self.value
        requires_grad = self.requires_grad
        if isinstance(power, Variable):
            new_value = new_value**(power.value)
            requires_grad = requires_grad or power.requires_grad
        else:
            new_value = new_value**power
        
        dtype = type(new_value)

        if flamb.environ['compute_grad']:
            last_operation = PowerOperator(self, power)
            return Variable(new_value, dtype=dtype, requires_grad=requires_grad, last_operation=last_operation)
        else:
            self.__init__(new_value, dtype=dtype, requires_grad=False, last_operation=None)



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

    def __ge_(self, var):
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


    def backward(self, accumulated_grad=None):
        if flamb.environ['compute_grad']:
            if accumulated_grad == None: # vérifier le == None
                accumulated_grad = 1

            self.grad += accumulated_grad
            # revoir au dessus

            last_operation = self.last_operation
            if last_operation != None:
                variables = last_operation.get_variables()
                grads = last_operation.gradient()

                for var, grad in zip(variables, grads):
                    if isinstance(var, Variable) and var.requires_grad:
                        var.backward(accumulated_grad*grad)
        
        else:
            raise Exception("Cannot compute gradient in flamb.no_grad context")
            


    

