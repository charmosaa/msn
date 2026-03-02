import meshio
import numpy as np

def setup_fem_2d(mesh_file, c_val=1.0):
    # Read mesh
    mesh = meshio.read(mesh_file)

    puntos = mesh.points[:, :2] # coordinates without Z

    triangles = mesh.cells_dict["triangle"]
    num_nodes = len(puntos)
    num_triangles = len(triangles)
    print(f"Malla cargada: {num_nodes} nodos y {num_triangles} elementos triangulares.")

    # Global matrix
    A_global = np.zeros((num_nodes, num_nodes))

    # Reference gradients (constant for linear triangles)
    grad_N_ref = np.array([
        [-1.0, -1.0], 
        [ 1.0,  0.0], 
        [ 0.0,  1.0]
    ])
    
    # Ensamblaje loop
    for e in range(num_triangles):
        nodes_tri = triangles[e]
        
        # Coordinates of the triangle vertices
        x1, y1 = puntos[nodes_tri[0]]
        x2, y2 = puntos[nodes_tri[1]]
        x3, y3 = puntos[nodes_tri[2]]
        
        # Calculate Jacobian matrix B_K
        B_K = np.array([
            [x2 - x1, x3 - x1],
            [y2 - y1, y3 - y1]
        ])
        
        # Determinant and Area
        det_B_K = np.linalg.det(B_K)
        area = abs(det_B_K) / 2.0
        
        # Inversa transpuesta del Jacobiano para pasar gradientes al dominio real
        inv_B_K_T = np.linalg.inv(B_K).T
        
        # Gradientes reales de este elemento específico
        grad_N_real = np.zeros((3, 2))
        for i in range(3):
            grad_N_real[i] = inv_B_K_T @ grad_N_ref[i]
            
        # Matrices Locales 3x3
        A_local = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                # Matriz de Esfuerzos (Stiffness/Difusión): w_ij = int(grad(Ni) * grad(Nj))
                W_ij = np.dot(grad_N_real[i], grad_N_real[j]) * area
                
                # Matriz de Masa (Reacción): m_ij = int(Ni * Nj)
                # Integración exacta para triángulos lineales:
                if i == j:
                    M_ij = area / 6.0  # diagonal
                else:
                    M_ij = area / 12.0 # fuera de la diagonal
                
                # Combinamos para la ecuación: A = W + c*M
                A_local[i, j] = W_ij + c_val * M_ij
                
        # 5. Ensamblaje Global
        for i in range(3):
            global_i = nodes_tri[i]
            for j in range(3):
                global_j = nodes_tri[j]
                A_global[global_i, global_j] += A_local[i, j]
                
    return A_global, puntos

if __name__ == '__main__':
    A, puntos = setup_fem_2d("mesh.msh", c_val=1.0)
    print(f"Dimensiones de la matriz global A: {A.shape}")