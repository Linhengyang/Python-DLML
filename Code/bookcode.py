
# Oct, 18, 2020
# https://d2l.ai/chapter_preliminaries/ndarray.html
import torch
x = torch.arange(12)
x.shape
x.numel()
X = x.reshape(-1, 4)
torch.zeros(2, 3, 4)
torch.ones(2, 3, 4)
torch.randn(3, 4)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# elementwise operation
# + - * / ** elementwise opeartion on any two tensors of the same shape
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x+y, x-y, x*y, x/y, x**y

torch.exp(x)

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim = 0)
torch.cat((X, Y), dim = 1)
X == Y
X.sum()

# broadcasting mechanism
a = torch.arange(6).reshape(2, 3, 1)
b = torch.arange(2).reshape(1, 2)
a, b
a+b # a replicates its columns and b replicates its rows

# indexing and slicing
X
X[-1]
X[1:3]
X[1, 2] = 9
X[0:2, :] = 12
X

#memory
before = id(Y)
Y = Y + X
id(Y) == before

Z = torch.zeros_like(Y)
Z
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))

before = id(X)
X += Y
id(X) == before

# conversion to other python objects
A = X.numpy()
A
B = torch.tensor(A)
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

# Oct 19, 2020
# https://d2l.ai/chapter_preliminaries/pandas.html

import os
def mkdir_if_not_exist(path):
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)

data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f: # need 
    f.write('NumRooms, Alley, Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')

import pandas as pd
data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na = True)
inputs

import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y

## Oct 23, 2020
import torch
x = torch.tensor([3.0])
y = torch.tensor([2.0])
x = torch.arange(4)
x.shape
A = torch.arange(20, dtype=float32).reshape(5,4)
A.T
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

X = torch.arange(24).reshape(2, 3, 4)
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone()
## hadamard product of vector-vector, matrix-matrix, elementwise
A*B

x = torch.arange(4, dtype=torch.float32)
x.sum()

## dot product of vector-vector
torch.dot()
## product of matrix-vector -- a transformation
A
x
torch.mv(A, x)
## product of matrix-matrix
A
B = torch.ones(4, 3)
torch.mm(A, B)

## norms
u = torch.tensor([3.0, -4.0])
torch.norm(u) #L2 norm
torch.abs(u).sum() #L1 norm
torch.norm(torch.ones(4,9)) # F norm of matrix

# derivatives
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h = {h:.5f}, numerical limits = {numerical_lim(f, 1, h): .5f}')
    h = h*0.1

# Auto Differentiation
import torch
x = torch.arange(4, dtype=torch.float32)
x.requires_grad_(True)
x.grad
y = 2*torch.dot(x, x)
y.backward()
x.grad
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

# Backward for Non-Scalar Variables is a matrix of derivatives
# here we only compute the sum of y
y = x * x
y.sum().backward()
## or passing a gradient as an argument
y.backward(torch.ones(len(x)))
x.grad

# Detaching Computation
x
x.grad
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z
z.sum().backward()
x.grad
# then get the grad of y w.r.t x
x.grad.zero_()
y.sum().backward()
x.grad

# Computing the gradient of Python Control Flow
import torch
a = torch.randn(size=(), requires_grad=True)
def f(a): # piecewise linear w.r.t input a
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
d = f(a)
d.backward()
a.grad == d / a

# Documentation
import torch
print(dir(torch.distributions))
