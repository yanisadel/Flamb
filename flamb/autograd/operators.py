from flamb.utils import *
import math

class BaseOperator:
    """
    Class defining an operation, allowing to remember of some operations (sum, product...)
    and to calculate gradients of the operation with respect to the parameters
    """
    def __init__(self, *variables):
        self.variables = list(variables)
    
    def gradient(self):
        """
        Returns the gradient with respect to the parameters
        """
        raise Exception("This function needs to be implemented")

    def get_variables(self):
        return self.variables



class SumOperator(BaseOperator):
    def gradient(self):
        return [1 for _ in range(len(self.variables))]
        

class ProductOperator(BaseOperator):
    def gradient(self):
        # Convert variables to int/float values
        variables = convert_variable_list(self.variables)

        grads = []
        n = len(variables)

        if 0 in variables: # revoir si ça fonctionne bien vu que ce sont des Variable
            for i in range(n):
                # All values different than 0 have gradient equal to 0
                if variables[i] == 0:
                    product = 1
                    for j in range(n):
                        if j != i:
                            product *= variables[j]
                    grads.append(product)
                else:
                    grads.append(0)

        else:
            product = 1
            for value in variables:
                product *= value
            
            for value in variables:
                grads.append(product/value)

        return grads



class DivisionOperator(BaseOperator):
    def gradient(self):
        # Convert variables to int/float values
        try:
            assert len(self.variables) == 2
        except:
            raise Exception("Cannot handle division with more than 2 variables")

        variables = convert_variable_list(self.variables)
        value1, value2 = variables

        grads = [1/value2, -value1/(value2**2)]
        return grads


class PowerOperator(BaseOperator):
    def __init__(self, variable, power):
        self.power = power
        self.variables = [variable]

    def gradient(self):
        variable = convert_variable(self.variables[0])
        power = convert_variable(self.power)
        return [power*(variable**(power-1))]


class ExpOperator(BaseOperator):
    def gradient(self):
        variable = convert_variable(self.variables[0])
        return [math.exp(variable)]

class CosOperator(BaseOperator):
    def gradient(self):
        variable = convert_variable(self.variables[0])
        return [-math.sin(variable)]

class SinOperator(BaseOperator):
    def gradient(self):
        variable = convert_variable(self.variables[0])
        return [math.cos(variable)]

class TanOperator(BaseOperator):
    def gradient(self):
        variable = convert_variable(self.variables[0])
        return [1 + math.tan(variable)**2]

class TanhOperator(BaseOperator):
    def gradient(self):
        variable = convert_variable(self.variables[0])
        return [1 - math.tanh(variable)**2]
        