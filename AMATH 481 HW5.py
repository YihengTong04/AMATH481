#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.sparse import spdiags
from scipy.linalg import lu, solve_triangular

# ---------------------- Part a ----------------------
# Parameters
time_span = np.arange(0, 4.5, 0.5) 
viscosity = 0.001  
domain_x, domain_y = 20, 20  
grid_x, grid_y = 64, 64  
total_points = grid_x * grid_y

x_vals = np.linspace(-domain_x / 2, domain_x / 2, grid_x + 1)
x_domain = x_vals[:grid_x]
y_vals = np.linspace(-domain_y / 2, domain_y / 2, grid_y + 1)
y_domain = y_vals[:grid_y]
X, Y = np.meshgrid(x_domain, y_domain)
vorticity_initial = np.exp(-X ** 2 - Y ** 2 / 20).flatten()

# Define spectral k values
kx_vals = (2 * np.pi / domain_x) * np.concatenate((np.arange(0, grid_x / 2), np.arange(-grid_x / 2, 0)))
kx_vals[0] = 1e-6 
ky_vals = (2 * np.pi / domain_y) * np.concatenate((np.arange(0, grid_y / 2), np.arange(-grid_y / 2, 0)))
ky_vals[0] = 1e-6
kx_mesh, ky_mesh = np.meshgrid(kx_vals, ky_vals)
k_squared = kx_mesh ** 2 + ky_mesh ** 2

length = domain_x 
point_count = total_points  
grid_count = grid_x  
delta = length / grid_count  

e0 = np.zeros((point_count, 1))  
e1 = np.ones((point_count, 1))  
e2 = np.copy(e1) 
e4 = np.copy(e0)  

#  Matrix A 
for idx in range(1, grid_count + 1):
    e2[grid_count * idx - 1] = 0  
    e4[grid_count * idx - 1] = 1 

e3 = np.zeros_like(e2)
e3[1:point_count] = e2[:point_count - 1]
e3[0] = e2[point_count - 1]

e5 = np.zeros_like(e4)
e5[1:point_count] = e4[:point_count - 1]
e5[0] = e4[point_count - 1]

# diagonal elements 
diag_A = [e1.flatten(), e1.flatten(), e5.flatten(),
          e2.flatten(), -4 * e1.flatten(), e3.flatten(),
          e4.flatten(), e1.flatten(), e1.flatten()]
offset_A = [-(point_count - grid_count), -grid_count, -grid_count + 1, -1, 0, 1, grid_count - 1, grid_count, (point_count - grid_count)]

matrix_A = spdiags(diag_A, offset_A, point_count, point_count) / (delta ** 2)

# Matrix B
diag_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offset_B = [-(point_count - grid_count), -grid_count, grid_count, (point_count - grid_count)]
matrix_B = spdiags(diag_B, offset_B, point_count, point_count) / (2 * delta)

# Matrix C
for idx in range(1, point_count):
    e1[idx] = e4[idx - 1]

diag_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offset_C = [-grid_count + 1, -1, 1, grid_count - 1]
matrix_C = spdiags(diag_C, offset_C, point_count, point_count) / (2 * delta)

matrix_A = matrix_A.toarray()
matrix_B = matrix_B.toarray()
matrix_C = matrix_C.toarray()

matrix_A[0, 0] = 2 / (delta ** 2)

# Define Right Hand Side
def compute_rhs(t, vorticity, grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, k_squared, viscosity):
    w = vorticity.reshape((grid_x, grid_y))
    fft_w = fft2(w)
    psi_fft = - fft_w / k_squared
    psi_real = np.real(ifft2(psi_fft)).reshape(total_points)
    rhs_val = (viscosity * np.dot(matrix_A, vorticity)
               - np.dot(matrix_B, psi_real) * np.dot(matrix_C, vorticity)
               + np.dot(matrix_B, vorticity) * np.dot(matrix_C, psi_real)
               )
    return rhs_val


solution = solve_ivp(compute_rhs, [0, 4], vorticity_initial, t_eval=time_span,
                     args=(grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, k_squared, viscosity), method="RK45")
result = solution.y
A1 = result
print('A1: \n', A1)


# In[15]:


# ---------------------- Part b ----------------------
# Method A\b
def compute_rhs_direct(t, vorticity, grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, viscosity):
    psi = np.linalg.solve(matrix_A, vorticity)
    rhs_direct = (
        viscosity * np.dot(matrix_A, vorticity)
        - np.dot(matrix_B, psi) * np.dot(matrix_C, vorticity)
        + np.dot(matrix_B, vorticity) * np.dot(matrix_C, psi)
    )
    return rhs_direct

solution_direct = solve_ivp(
    compute_rhs_direct, [0, 4], vorticity_initial, t_eval=time_span,
    args=(grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, viscosity), method="RK45"
)
A2 = solution_direct.y
print('A2 (Direct Method): \n', A2)

# Method LU decomposition
def compute_rhs_lu(t, vorticity, grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, viscosity):
    permuted_rhs = np.dot(P_lu, vorticity)
    intermediate_y = solve_triangular(L_lu, permuted_rhs, lower=True)
    psi = solve_triangular(U_lu, intermediate_y)
    rhs_lu = (
        viscosity * np.dot(matrix_A, vorticity)
        - np.dot(matrix_B, psi) * np.dot(matrix_C, vorticity)
        + np.dot(matrix_B, vorticity) * np.dot(matrix_C, psi)
    )
    return rhs_lu

# Perform LU decomposition of matrix_A
P_lu, L_lu, U_lu = lu(matrix_A)

solution_lu = solve_ivp(
    compute_rhs_lu, [0, 4], vorticity_initial, t_eval=time_span,
    args=(grid_x, grid_y, total_points, matrix_A, matrix_B, matrix_C, viscosity), method="RK45"
)
A3 = solution_lu.y
print('A3 (LU Decomposition): \n', A3)


print('Shape of A2:', A2.shape)
print('Shape of A3:', A3.shape)


# In[ ]:




