#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_ivp, simpson
import math


def shooting_equation(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]


tolerance = 1e-6               
color_list = ['r', 'b', 'g', 'c', 'm'] 
domain_limit = 4         
delta_x = 0.1          
x_values = np.arange(-domain_limit, domain_limit + delta_x, delta_x)  
epsilon_start = 0.1      


A1 = np.zeros((len(x_values), 5)) 
A2 = np.zeros(5)                 


for mode in range(1, 6):
    epsilon = epsilon_start 
    depsilon = 0.2       


    for _ in range(1000):
        initial_conditions = [1, np.sqrt(domain_limit**2 - epsilon)]
        
        solution = solve_ivp(
            shooting_equation,
            [x_values[0], x_values[-1]],
            initial_conditions,
            args=(epsilon,),
            t_eval=x_values
        )

        boundary_condition = solution.y[1, -1] + np.sqrt(domain_limit**2 - epsilon) * solution.y[0, -1]
        
        if abs(boundary_condition) < tolerance:
            break

        if ((-1) ** (mode + 1) * boundary_condition) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2

    A2[mode - 1] = epsilon
    epsilon_start = epsilon + 0.1 

    normalization = np.trapz(solution.y[0] ** 2, x_values)
    eigenfunction = np.abs(solution.y[0] / np.sqrt(normalization))
    A1[:, mode - 1] = eigenfunction
    plt.plot(x_values, eigenfunction, color_list[mode - 1], label=f"Mode {mode}")

plt.xlabel("x")
plt.ylabel("Eigenfunction")
plt.title("Eigenfunctions for the First Five Modes")
plt.legend()
plt.show()

print("A1:")
print(A1)
print("A2:")
print(A2)


# In[63]:


#part b
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

length = 4
delta = 0.1
x_vals = np.arange(-length, length + delta, delta)
num_points = len(x_vals) - 2

matrix_A = np.zeros((num_points, num_points))
for i in range(num_points):
    matrix_A[i, i] = -2 - (x_vals[i + 1] ** 2) * (delta ** 2)

for i in range(num_points - 1):
    matrix_A[i, i + 1] = 1
    matrix_A[i + 1, i] = 1

matrix_B = matrix_A.copy()

boundary_matrix_start = np.zeros((num_points, num_points))
boundary_matrix_start[0, 0] = 4 / 3
boundary_matrix_start[0, 1] = -1 / 3

boundary_matrix_end = np.zeros((num_points, num_points))
boundary_matrix_end[-1, -2] = -1 / 3
boundary_matrix_end[-1, -1] = 4 / 3

matrix_total = matrix_B + boundary_matrix_start + boundary_matrix_end
matrix_total /= delta ** 2

eigenvalues, eigenvectors = eigs(-matrix_total, k=5, which='SM')

phi_start = (4 / 3) * eigenvectors[0, :] - (1 / 3) * eigenvectors[1, :]
phi_end = - (1 / 3) * eigenvectors[-2, :] + (4 / 3) * eigenvectors[-1, :]

eigenvectors_full = np.vstack((phi_start, eigenvectors, phi_end))

for idx in range(5):
    norm_const = np.trapz(eigenvectors_full[:, idx] ** 2, x_vals)
    eigenvectors_full[:, idx] = np.abs(eigenvectors_full[:, idx] / np.sqrt(norm_const))
    plt.plot(x_vals, eigenvectors_full[:, idx])

plt.legend(
    [r"$\phi_1$", r"$\phi_2$", r"$\phi_3$", r"$\phi_4$", r"$\phi_5$"],
    loc="upper right"
)
plt.show()

A3 = eigenvectors_full
A4 = eigenvalues

print("A3:")
print(A3)
print("A4:")
print(A4)


# In[61]:


#part c
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson

def shooting_equation(x, y, epsilon, gamma):
    return [
        y[1],
        (gamma * y[0] ** 2 + x ** 2 - epsilon) * y[0]
    ]

tolerance = 1e-6
domain_limit = 2
delta_x = 0.1
x_range = np.arange(-domain_limit, domain_limit + delta_x, delta_x)
gamma_list = [0.05, -0.05] 

eigenfuncs_gamma_pos = np.zeros((len(x_range), 2))
eigenfuncs_gamma_neg = np.zeros((len(x_range), 2))
A6, A8 = np.zeros(2), np.zeros(2)  

for gamma in gamma_list:
    epsilon_initial = 0.1
    amplitude = 1e-6

    for mode in range(1, 3):
        delta_amplitude = 0.01

        for _ in range(100):
            epsilon = epsilon_initial
            delta_epsilon = 0.2

            for _ in range(100):
                y0 = [amplitude, np.sqrt(domain_limit ** 2 - epsilon) * amplitude]

                solution = solve_ivp(
                    lambda x, y: shooting_equation(x, y, epsilon, gamma),
                    [x_range[0], x_range[-1]],
                    y0,
                    t_eval=x_range
                )
                y_sol = solution.y.T
                x_sol = solution.t

                boundary_condition = y_sol[-1, 1] + np.sqrt(domain_limit ** 2 - epsilon) * y_sol[-1, 0]
                if abs(boundary_condition) < tolerance:
                    break
                    
                if (-1) ** (mode + 1) * boundary_condition > 0:
                    epsilon += delta_epsilon
                else:
                    epsilon -= delta_epsilon
                    delta_epsilon /= 2

            norm = simpson(y_sol[:, 0] ** 2, x=x_sol)
            if abs(norm - 1) < tolerance:
                break

            if norm < 1:
                amplitude += delta_amplitude
            else:
                amplitude -= delta_amplitude
                delta_amplitude /= 2

        epsilon_initial = epsilon + 0.2

        if gamma > 0:
            eigenfuncs_gamma_pos[:, mode - 1] = np.abs(y_sol[:, 0])
            A6[mode - 1] = epsilon
        else:
            eigenfuncs_gamma_neg[:, mode - 1] = np.abs(y_sol[:, 0])
            A8[mode - 1] = epsilon
            
plt.plot(x_range, eigenfuncs_gamma_pos)
plt.plot(x_range, eigenfuncs_gamma_neg)
plt.legend([r"$\phi_1$", r"$\phi_2$"], loc="upper right")
plt.xlabel("x")
plt.ylabel("Eigenfunction")
plt.title("Eigenfunctions for Different Gamma Values")
plt.show()

A5 = eigenfuncs_gamma_pos
A7 = eigenfuncs_gamma_neg

print("A6:")
print(A6)
print("A8:")
print(A8)

#part d
import numpy as np
from scipy.integrate import solve_ivp

def differential_system(x, y, energy):
    return [y[1], (x**2 - energy) * y[0]]

length = 2
x_interval = [-length, length]
energy = 1
amplitude = 1
initial_conditions = [amplitude, np.sqrt(length ** 2 - energy) * amplitude]
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

avg_dt_RK45 = []
avg_dt_RK23 = []
avg_dt_Radau = []
avg_dt_BDF = []

for tol in tolerances:
    options = {'rtol': tol, 'atol': tol}

    sol_RK45 = solve_ivp(differential_system, x_interval, initial_conditions, method='RK45', args=(energy,), **options)
    sol_RK23 = solve_ivp(differential_system, x_interval, initial_conditions, method='RK23', args=(energy,), **options)
    sol_Radau = solve_ivp(differential_system, x_interval, initial_conditions, method='Radau', args=(energy,), **options)
    sol_BDF = solve_ivp(differential_system, x_interval, initial_conditions, method='BDF', args=(energy,), **options)

    avg_dt_RK45.append(np.mean(np.diff(sol_RK45.t)))
    avg_dt_RK23.append(np.mean(np.diff(sol_RK23.t)))
    avg_dt_Radau.append(np.mean(np.diff(sol_Radau.t)))
    avg_dt_BDF.append(np.mean(np.diff(sol_BDF.t)))

fit_RK45 = np.polyfit(np.log(avg_dt_RK45), np.log(tolerances), 1)
fit_RK23 = np.polyfit(np.log(avg_dt_RK23), np.log(tolerances), 1)
fit_Radau = np.polyfit(np.log(avg_dt_Radau), np.log(tolerances), 1)
fit_BDF = np.polyfit(np.log(avg_dt_BDF), np.log(tolerances), 1)

slope_RK45 = fit_RK45[0]
slope_RK23 = fit_RK23[0]
slope_Radau = fit_Radau[0]
slope_BDF = fit_BDF[0]

A9 = np.array([slope_RK45, slope_RK23, slope_Radau, slope_BDF])

print("A9:")
print(A9)


# In[64]:


#part e
import numpy as np
import math
from scipy.integrate import simpson

def hermite_poly_0(x):
    return np.ones_like(x)

def hermite_poly_1(x):
    return 2 * x

def hermite_poly_2(x):
    return 4 * x**2 - 2

def hermite_poly_3(x):
    return 8 * x**3 - 12 * x

def hermite_poly_4(x):
    return 16 * x**4 - 48 * x**2 + 12

length = 4
delta_x = 0.1
x_values = np.arange(-length, length + delta_x, delta_x)

hermite_matrix = np.column_stack([
    hermite_poly_0(x_values),
    hermite_poly_1(x_values),
    hermite_poly_2(x_values),
    hermite_poly_3(x_values),
    hermite_poly_4(x_values)
])

phi = np.zeros_like(hermite_matrix)

for n in range(5):
    phi[:, n] = (
        np.exp(-x_values**2 / 2) * hermite_matrix[:, n]
    ) / np.sqrt(math.factorial(n) * (2 ** n) * np.sqrt(np.pi))

error_func_a = np.zeros(5)
error_func_b = np.zeros(5)
error_eigen_a = np.zeros(5)
error_eigen_b = np.zeros(5)

for n in range(5):
    error_func_a[n] = simpson((np.abs(A1[:, n]) - np.abs(phi[:, n])) ** 2, x=x_values)
    error_func_b[n] = simpson((np.abs(A3[:, n]) - np.abs(phi[:, n])) ** 2, x=x_values)
    
    exact_eigenvalue = 2 * (n + 1) - 1
    error_eigen_a[n] = 100 * np.abs(A2[n] - exact_eigenvalue) / exact_eigenvalue
    error_eigen_b[n] = 100 * np.abs(A4[n] - exact_eigenvalue) / exact_eigenvalue

A10 = error_func_a
A11 = error_eigen_a

A12 = error_func_b
A13 = error_eigen_b

print("A10:")
print(A10)
print("A11:")
print(A11)

print("A12:")
print(A12)
print("A13:")
print(A13)


# In[ ]:




