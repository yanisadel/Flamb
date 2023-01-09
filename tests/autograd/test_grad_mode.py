import flamb
from flamb import Variable


def test_no_grad():
    x = Variable(4, requires_grad=True)
    with flamb.no_grad():
        assert flamb.environ["is_grad_enabled"] == False
        y = x ** 2

    assert y.requires_grad == False, "y.requires_grad must be False"
    assert (
        flamb.environ["is_grad_enabled"] == True
    ), "flamb.environ['is_grad_enabled'] should has turned back to True"


if __name__ == "__main__":
    test_no_grad()
