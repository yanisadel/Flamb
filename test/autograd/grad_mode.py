import flamb
from flamb import Variable

def test_no_grad():
    x = Variable(4, requires_grad=True)
    with flamb.no_grad():
        assert flamb.environ['compute_grad'] == False
        y = x**2


if __name__ == '__main__':
    test_no_grad()
