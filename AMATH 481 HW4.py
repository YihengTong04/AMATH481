#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

m = 8  
n = m * m  
L = 10
delta = (2 * L) / m

e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

# Matrix A
for j in range(1, m + 1):
    e2[m * j - 1] = 0 
    e4[m * j - 1] = 1 

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n - 1]
e3[0] = e2[n - 1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n - 1]
e5[0] = e4[n - 1]


diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(),
               e2.flatten(), -4 * e1.flatten(), e3.flatten(),
               e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(n - m), -m, -m + 1, -1, 0, 1, m - 1, m, (n - m)]


A = spdiags(diagonals_A, offsets_A, n, n).toarray() / (delta ** 2)

#Matrix B
diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
B = spdiags(diagonals_B, offsets_B, n, n).toarray() / (delta * 2)

#Matrix C
for i in range(1, n):
    e1[i] = e4[i - 1]
diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,  m - 1]
C = spdiags(diagonals_C, offsets_C, n, n).toarray() / (delta * 2)

A1 = A
A2 = B
A3 = C

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C:\n", C)

