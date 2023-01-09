import flamb


def convert_variable(var):
    """
    Convert a flamb.Variable into a value (integer or float). 
    Also works if var is an int or a float
    """
    if isinstance(var, flamb.Variable):
        return var.value

    elif isinstance(var, (int, float)):
        return var

    else:
        raise Exception(f"Cannot convert {type(var)} to a value")


def convert_variable_list(l):
    """
    Works the same way as convert_variable but for a list of variables
    """
    return [convert_variable(var) for var in l]
