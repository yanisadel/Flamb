import flamb

def convert_variable(var):
    """
    Convert a Variable into a value (integer or float). Also works if var is not a variable
    """
    if isinstance(var, flamb.Variable):
        return var.value

    elif isinstance(var, (int, float)):
        return var
    
    else:
        raise Exception(f"Cannot convert {type(var)} to a value")

def convert_variable_list(l):
    return [convert_variable(var) for var in l]