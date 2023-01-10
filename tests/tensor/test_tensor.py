import flamb
from flamb import Tensor
from copy import deepcopy

l_ref = [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]


def test_shape():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    assert l.shape == (2, 2, 4), f"Shape is {l.shape} but should be (2,2,4)"


def test_read_value():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    assert l[1][1][2] == 15, f"l[1][1][2] should be 15, but it is {l[1][1][2]}"

    assert l[(1, 1, 2)] == 15, f"l[(1,1,2)] should be 15, but it is {l[(1,1,2)]}"


def test_modify_value():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    l[(0, 1, 2)] = 50
    assert (
        l[(0, 1, 2)] == 50
    ), f"l[0][1][2] should be have been modified to 50, but it is {l[0][1][2]}"

    l = deepcopy(l_ref)
    l = Tensor(l)
    l[0][1][2] = 50
    # assert (l[0][1][2] == 50), f"l[0][1][2] should be have been modified to 50, but it is {l[0][1][2]}"


def test_loop_on_indicies():
    shape = (2, 2)
    l = [index for index in Tensor._loop_on_indicies(shape)]

    assert l == [(0, 0), (0, 1), (1, 0), (1, 1)], "The loop on indicies does not work"

    shape = (2, 2, 2)
    l = [index for index in Tensor._loop_on_indicies(shape)]

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


def test_product_operator():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    l2 = 4 * l
    assert (
        l2[(0, 1, 2)] == l_ref[0][1][2] * 4
    ), f"l[(0,1,2)] should be have been equal to {l_ref[0][1][2]*2}, but it is {l2[(0,1,2)]}"


def test_sum_operator():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    l2 = l + l
    assert (
        l2[(0, 1, 2)] == l_ref[0][1][2] * 2
    ), f"l[(0,1,2)] should be have been equal to {l_ref[0][1][2]*2}, but it is {l2[(0,1,2)]}"


def test_sub_operator():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)
    l2 = Tensor(
        [[[1, 2 + 11, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]
    )

    l3 = l2 - l
    assert (
        l3[(0, 0, 1)] == 11
    ), f"l3[(0,0,1)] should be have been equal to {11}, but it is {l3[(0, 0, 1)]}"


def test_sum_method():
    global l_ref
    l = deepcopy(l_ref)
    l = Tensor(l)

    assert l.sum() == 16 * 17 // 2, "Sum method is not correct"


if __name__ == "__main__":
    test_shape()
    test_read_value()
    test_modify_value()
    test_loop_on_indicies()
    test_product_operator()
    test_sum_operator()
    test_sub_operator()
    test_sum_method()
