def convert_variable(var):
    """
    Convert a Variable into a value (integer or float). Also works if var is not a variable
    """
    try:
        res = var.value
        return res

    except:
        return var

def convert_variable_list(l):
    return [convert_variable(var) for var in l]