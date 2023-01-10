import flamb
from flamb import nn


def test_module():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 30)
            self.layer2 = nn.Linear(30, 50)

        def forward(self, x):
            x = self.layer(x)
            return self.layer2(x)

        
    x = flamb.ones((32, 10))
    model = Model()

    assert model.parameters is None, f"model.parameters should be None but is {model.parameters}"
    output = model(x)

    assert output.shape == (32, 50), f"Ouput shape should be (32, 50) but it is {output.shape}"

    assert model.parameters.shape == (30*10 + 30 + 50*30 + 50,), "The number of parameters is not correct"


if __name__ == '__main__':
    test_module()