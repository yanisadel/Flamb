from flamb.tensor.utils import loop_on_indicies

def test_loop_on_indicies():
    shape = (2, 2)
    l = [index for index in loop_on_indicies(shape)]
    assert l == [(0, 0), (0, 1), (1, 0), (1, 1)], "The loop on indicies does not work"

    shape = (2, 2, 2)
    l = [index for index in loop_on_indicies(shape)]
    assert l == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ], "The loop on indicies does not work"
    pass


if __name__ == '__main__':
    test_loop_on_indicies()