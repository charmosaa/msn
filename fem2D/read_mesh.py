import meshio
import numpy as np

def setup_fem_2d(mesh_file, c_val=1.0):
    # 1. Leer la malla
    mesh = meshio.read(mesh_file)
    
    # Extraer coordenadas de los nodos (ignoramos Z porque es 2D)
    puntos = mesh.points[:, :2] 
    
    # Extraer la conectividad de los triángulos (nodos que forman cada elemento)
    # Gmsh guarda varios tipos de celdas (líneas, triángulos), filtramos por "triangle"
    celdas = mesh.cells_dict["triangle"]
    
    num_nodos = len(puntos)
    num_elementos = len(celdas)
    
    print(f"Malla cargada: {num_nodos} nodos y {num_elementos} elementos triangulares.")

    # 2. Inicializar Matriz Global
    # Usamos una matriz densa para el ejemplo, pero en problemas grandes 
    # se usan matrices dispersas (scipy.sparse)
    A_global = np.zeros((num_nodos, num_nodos))
    
    # 3. Gradientes en el elemento de referencia (constantes)
    # grad(N1) = [-1, -1], grad(N2) = [1, 0], grad(N3) = [0, 1]
    grad_N_ref = np.array([
        [-1.0, -1.0], 
        [ 1.0,  0.0], 
        [ 0.0,  1.0]
    ])
    
    # 4. Bucle de Ensamblaje (Iteramos sobre cada triángulo)
    for e in range(num_elementos):
        nodos_elem = celdas[e] # Índices de los 3 nodos del triángulo 'e'
        
        # Coordenadas (x,y) de los 3 vértices
        x1, y1 = puntos[nodos_elem[0]]
        x2, y2 = puntos[nodos_elem[1]]
        x3, y3 = puntos[nodos_elem[2]]
        
        # Calcular la Matriz Jacobiana B_K
        B_K = np.array([
            [x2 - x1, x3 - x1],
            [y2 - y1, y3 - y1]
        ])
        
        # Determinante y Área
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
            global_i = nodos_elem[i]
            for j in range(3):
                global_j = nodos_elem[j]
                A_global[global_i, global_j] += A_local[i, j]
                
    return A_global, puntos

if __name__ == '__main__':
    A, puntos = setup_fem_2d("mesh.msh", c_val=1.0)
    print(f"Dimensiones de la matriz global A: {A.shape}")