import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.sin(x)

def u_initial(x):
    return np.cos(x)

def create_mesh(distance, nodes):
    return np.linspace(0, distance, nodes)

def assemble_base_matrices(L, N):
    x = create_mesh(L, N)
    h = x[1] - x[0]

    # reference points
    gauss_points = [(1 - 1/np.sqrt(3))/2, (1 + 1/np.sqrt(3))/2] 
    weight = 0.5

    # matrices and vector
    M = np.zeros((N, N))
    W = np.zeros((N, N))
    B = np.zeros(N)

    w_val  = 1.0 / h        # stiffness contribution from -u''
    m_diag = h / 3.0        # diagonal part for mass matrix
    m_off  = h / 6.0        # off-diagonal part for mass matrix

    for i in range(N - 1):
        # matrix W
        W[i, i]     += w_val
        W[i, i+1]   -= w_val
        W[i+1, i]   -= w_val
        W[i+1, i+1] += w_val

        # matrix M
        M[i, i]     += m_diag
        M[i, i+1]   += m_off
        M[i+1, i]   += m_off
        M[i+1, i+1] += m_diag

        # vector B
        for gp in gauss_points:
            xi = x[i] + gp * h                          # map to physical element   
            B[i]   += f(xi) * weight * h * (1 - gp)     # weight for node i
            B[i+1] += f(xi) * weight * h * gp           # weight for node i+1          

    return M, W, B, x

def solve_fem_1d_evolutive_dd(L, N, k_param, T, dt, u0_bc, uL_bc):
    M, W, B, x = assemble_base_matrices(L, N)

    M[0, :] = 0 
    W[0, :] = 0
    W[0, 0] = 1
    B[0] = k_param * u0_bc

    M[-1, :] = 0               
    W[-1, :] = 0               
    W[-1, -1] = 1              
    B[-1] = k_param * uL_bc
    
    num_steps = int(T / dt)
    u_prev = u_initial(x) 
    u_prev[0] = u0_bc   
    u_prev[-1] = uL_bc  
    u_history = [u_prev.copy()]

    A_time = M + dt * k_param * W

    for step in range(1, num_steps + 1):
        R_time = M.dot(u_prev) + dt * B

        u_curr = np.linalg.solve(A_time, R_time)
        
        u_prev = u_curr.copy()
        u_history.append(u_curr)
        
    return x, u_curr, u_history

if __name__ == '__main__':
    L_val = 10
    x_plot, u_final, history = solve_fem_1d_evolutive_dd(L_val, N=100, k_param=1.0, T=5.0, dt=0.1, u0_bc=1.0, uL_bc=-1.0)

    plt.figure(figsize=(10, 6))

    for i, u_t in enumerate(history):
        plt.plot(x_plot, u_t)

    plt.title("Evolutive fem1D solution for $du/dt - k u'' = f$")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.grid(True)
    plt.show()