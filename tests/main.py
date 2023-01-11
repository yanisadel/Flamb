"""
Test of a model that would learn to calculate something
"""

"""
Il faut que je corrige
- le fait que flamb.no_grad crée des variables avec requires_grad = False, donc si je modifie une variable dans ce contexte, cette variable n'est plus traquée
"""

import flamb
from flamb import Variable
from flamb import functional as F

"""
Try to minimize the square function
"""


def f(x):
    return x**2

x = Variable(5, requires_grad=True)
learning_rate = 1e-2
params = flamb.to_tensor([x])
optimizer = flamb.nn.optimizers.SGD(params, learning_rate=learning_rate)

for _ in range(500):        
    loss = f(x)
    print(f"Loss : {loss}")
    loss.backward()
    optimizer.step()

print(x)




def f(x):
    return x**2

x = Variable(5)
learning_rate = 1e-2

for _ in range(500):        
    loss = f(x)
    loss.backward()
    with flamb.no_grad():
        print(f"Loss : {loss}")
        x -= learning_rate*x.grad
        x.grad = 0
    x.requires_grad = True

print(x)




"""
Try to minimize the square function with two parameters
"""

def g(x, y):
    return x**2 + (y - 3)**2

x = Variable(5)
y = Variable(2)
learning_rate = 1e-2

for _ in range(500):        
    loss = g(x, y)
    loss.backward()
    with flamb.no_grad():
        print(f"Loss : {loss}")
        x -= learning_rate*x.grad
        y -= learning_rate*y.grad
        x.grad = 0
        y.grad = 0
    x.requires_grad = True
    y.requires_grad = True

print(x, y)


"""
Try a more complex problem
"""
weight_list = [80, 75, 72, 40, 34, 50, 61, 54, 52]
size_list = [180, 170, 160, 150, 120, 149, 180, 142, 164]
imc_list = [24.69, 25.95, 28.12, 17.77, 23.6, 22.5, 18.8, 26.7, 19.3]
weight_list = [value for value in weight_list]
size_list = [value for value in size_list]
imc_list = [value for value in imc_list]


vars = [Variable(1e-2) for _ in range(6)]


learning_rate = 1e-2

def model(weight, size):
    x1 = F.tanh(weight*vars[0] + size*vars[1])
    x2 = F.tanh(weight*vars[2] + size*vars[3])
    x = vars[4]*x1 + vars[5]*x2
    return x

n = len(weight_list)
for _ in range(1000):
    loss = 0
    for i in range(n):
        loss += (imc_list[i] - model(weight_list[i], size_list[i]))**2
    loss /= n
    
    loss.backward()
    with flamb.no_grad():
        print(f"Loss : {loss}, vars : {vars}, grad : {vars[0].grad}")
        for i in range(6):
            vars[i] -= learning_rate*vars[i].grad
            vars[i].grad = 0
            vars[i].requires_grad = True
    
