"""
Test of a model that would learn to calculate something
"""
import flamb
from flamb import Variable

weight = [80, 75, 72, 40, 34, 50, 61, 54, 52]
size = [180, 170, 160, 150, 120, 149, 180, 142, 164]
imc = [24.69, 25.95, 28.12, 17.77, 23.6, 22.5, 18.8, 26.7, 19.3]

a = Variable(1)
b = Variable(1)


n = len(weight)
for i in range(1):
    res = weight[i]*a + size[i]*b
    loss = imc[i] - res
    loss.backward()



with flamb.no_grad():
    a -= a.grad
    b -= b.grad
    