import flamb
from flamb import nn
from flamb import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(500000)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 1)

    def forward(self, x):
        x = F.ReLU(self.layer1(x))
        x = F.ReLU(self.layer2(x))
        x = F.ReLU(self.layer3(x))
        return x

model = Model()

learning_rate = 1e-5
EPOCHS = 3

losses = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for _ in range(1):
        x = flamb.ones((16, 10))
        output = model(x)
        loss = (output.sum()/16)**2
        loss.backward()
        nb_parameters = len(model.parameters)
        with flamb.no_grad():
            for i in range(nb_parameters):
                model.parameters[i] -= learning_rate*model.parameters[i].grad
                model.parameters[i].requires_grad = True
                model.parameters[i].grad = 0
                model.parameters[i].last_operation = None
        total_loss += loss

    losses.append(total_loss.get_value())
    print(f"Loss : {total_loss}")

plt.plot([i for i in range(1, EPOCHS+1)], losses)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
