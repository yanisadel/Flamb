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
        self.initialize_parameters()

    def __call__(self, x):
        x = F.ReLU(self.layer1(x))
        x = F.ReLU(self.layer2(x))
        x = self.layer3(x)
        return x

    

model = Model()
loss_object = nn.losses.MSE()
optimizer = nn.optimizers.SGD(model.parameters, learning_rate=1e-4)

EPOCHS = 50

losses = []
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0
    for _ in range(1):
        x = flamb.ones((16, 10))
        output = model(x)
        target = flamb.zeros((16,1))
        loss = loss_object(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss

    losses.append(total_loss.get_value())
    print(f"Loss : {total_loss}")

plt.plot([i for i in range(1, EPOCHS+1)], losses)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
