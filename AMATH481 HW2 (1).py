#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
tol = 1e-6
col = ['r', 'b', 'g', 'c', 'm']  

k = 1 
beta_start = 0.1 
L = 4 
dx = 0.1
xshoot = np.arange(-L, L + dx, dx) 

def shoot2(y, x,beta):
    return [y[1], (x**2 - beta) * y[0]]

A1 = np.zeros((len(xshoot), 5))
A2 = np.zeros(5)
for modes in range(1, 6):  
    beta = beta_start 
    dbeta = 0.2
    
    for _ in range(1000):
        Y0 = [1, np.sqrt(L**2 - beta)]
        y = odeint(shoot2, Y0, xshoot, args=(beta, ))
        
        if abs(y[-1, 1] + np.sqrt(L**2 - beta) * y[-1, 0]) < tol:
            break
        
        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - beta) * y[-1, 0])) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2
    
    beta_start = beta + 0.1 
    A2[modes - 1] = beta
    norm = np.trapz(y[:, 0] ** 2, xshoot) 
    phi_normalized = abs(y[:, 0] / np.sqrt(norm))
    
    A1[:, modes - 1] =  phi_normalized
    
    plt.plot(xshoot,phi_normalized, col[modes-1])

plt.title("First 5 Normalized Eigenfunctions")
plt.xlabel("x")
plt.ylabel(r"$|\phi_n(x)|$")
plt.show()


print("EigenFunctions (A1):" , A1)
print("Eigenvalues (A2):", A2)


# In[ ]:




