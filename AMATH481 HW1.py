#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Question 1
import numpy as np
x = np.array([-1.6])
np.set_printoptions(precision=15)
for j in range(1000):
    x_new = x[j] - (x[j] * np.sin(3 * x[j]) - np.exp(x[j])) / (np.sin(3 * x[j]) + 3 * x[j] * np.cos(3 * x[j]) - np.exp(x[j]))
    x = np.append(x, x_new)
    fc = x[j] * np.sin(3 * x[j]) - np.exp(x[j])
    if abs(fc) < 1e-6:
        break
A1 = []
A1 = x
print("Steps needed for Newton's method:", [j + 1]) 
print(f"Root found at x = {x_new}")
print("A1 (Newton-Raphson x values):", A1)


# In[12]:


import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x * np.sin(3 * x) - np.exp(x)

xl = -0.7
xr = -0.4

dx = 0.1
x = np.arange(-1, 1 + dx, dx)
y = f(x)
plt.plot(x, y)
plt.axhline(0, color='black',linewidth=0.5)  
plt.axvline(0, color='black',linewidth=0.5)  
plt.title("f(x) = x*sin(3*x) - exp(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

A2 = []
for j in range(1000):
    xc = (xr + xl) / 2  
    fc = f(xc)          
    A2.append(xc)
    if fc > 0:
        xl = xc
    else:
        xr = xc


    if abs(fc) < 1e-6:
        break

print(f"Root found at x = {xc}")
print(f"f(x) = {fc}")
print(f"Total number of iterations: {j + 1}")


# In[13]:


A3 = [len(A1), len(A2)] 
print("A1 (Newton-Raphson x values):", A1)
print("A2 (Bisection Method x values):", A2)
print("A3 (Number of iterations for Newton and Bisection respectively):", A3)


# In[8]:


import numpy as np
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B

A5 = 3 * x - 4 * y

A6 = A @ x 

A7 = B @ (x - y)

A8 = D @ x

A9 = D @ y + z

A10 = A @ B

A11 = B @ C

A12 = C @ D


print("A4 = ", A4)
print("A5 = ", A5)
print("A6 = ", A6)
print("A7 = ", A7)
print("A8 = ", A8)
print("A9 = ", A9)
print("A10 = ", A10)
print("A11 = ", A11)
print("A12 = ", A12)


# In[ ]:




