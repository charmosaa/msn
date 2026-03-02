from math import sin

import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.sin(x) 

def create_mesh(distance, nodes):
    return np.linspace(0, distance, nodes)



def solve_fem_1d_dd(L, N, c, u0, uL):
    A, B, x = create_matrix_vector_mesh(L, N, c)

    # Dirichlet at x=0
    A[0, :] = 0
    A[0, 0] = 1
    B[0] = u0

    # Dirichlet at x=L
    A[-1, :] = 0
    A[-1, -1] = 1
    B[-1] = uL
    
    # solve the linear system
    u = np.linalg.solve(A, B)
    return x, u



def solve_fem_1d_nd(L, N, c, u0_prime, uL):
    A, B, x = create_matrix_vector_mesh(L, N, c)

    # last node (Dirichlet at x=L)
    A[-1, :] = 0
    A[-1, -1] = 1
    B[-1] = uL

    # Neumann at x=0
    B[0] += u0_prime

    # solve the linear system
    u = np.linalg.solve(A, B)
    return x, u



def solve_fem_1d_dn(L, N, c, u0, uL_prime):
    A, B, x = create_matrix_vector_mesh(L, N, c)

    # last node (Neumann at x=L)
    B[-1] += uL_prime

    # Dirichlet at x=0
    A[0, :] = 0
    A[0, 0] = 1
    B[0] = u0
    
    # solve the linear system
    u = np.linalg.solve(A, B)
    return x, u



def create_matrix_vector_mesh(L, N, c):
    x = create_mesh(L, N)
    h = x[1] - x[0]

    # reference points
    gauss_points = [(1 - 1/np.sqrt(3))/2, (1 + 1/np.sqrt(3))/2] 
    weight = 0.5

    # matrix and vector Au = B
    A = np.zeros((N, N))
    B = np.zeros(N)

    k_val   = 1.0 / h       # stiffness contribution from -u''
    m_diag  = c * h / 3.0   # diagonal part from +cu (the same index)
    m_off   = c * h / 6.0   # off-diagonal part from +cu  (different index)

    for i in range(N - 1):

        # slope part from -u'':
        A[i, i]     += k_val
        A[i, i+1]   -= k_val
        A[i+1, i]   -= k_val
        A[i+1, i+1] += k_val

        # part from +cu 
        A[i, i]     += m_diag
        A[i, i+1]   += m_off
        A[i+1, i]   += m_off
        A[i+1, i+1] += m_diag
        
        for gp in gauss_points:
            xi = x[i] + gp * h                          # map to physical element   

            B[i]   += f(xi) * weight * h * (1 - gp)     # weight for node i
            B[i+1] += f(xi) * weight * h * gp           # weight for node i+1

    return A, B, x



if __name__ == '__main__':  
    x_plot, u_sol = solve_fem_1d_dn(10, 10, 1, 1, 2)

    for i in range(10):
        print(f"x: {x_plot[i]:.2f} u: {u_sol[i]:.4f}")

    plt.plot(x_plot, u_sol, label='u(x) FEM')
    plt.title("Solution for $-u'' + cu = f$")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()
    plt.show()