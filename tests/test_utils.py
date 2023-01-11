import flamb
from flamb.utils import convert_variable, convert_variable_list


def test_convert_variable():
    """Test the function that converts a flamb.Variable to a scalar (int or float)"""
    var = flamb.Variable(4)
    converted = convert_variable(var)
    assert type(converted) == int and converted == 4

    var = flamb.Variable(4.4)
    converted = convert_variable(var)
    assert type(converted) == float and converted == 4.4

    var = 2
    converted = convert_variable(var)
    assert type(converted) == int and converted == 2

    var = 2.4
    converted = convert_variable(var)
    assert type(converted) == float and converted == 2.4


def test_convert_variable_list():
    """Test the convert_variable_list function"""
    l = [flamb.Variable(4), 2.5]
    assert convert_variable_list(l) == [4, 2.5]


if __name__ == '__main__':
    test_convert_variable()
    test_convert_variable_list()