import meshio
import numpy as np

grad_N_ref = np.array([
    [-1.0, -1.0], 
    [ 1.0,  0.0], 
    [ 0.0,  1.0]
])

def f(x, y):
    return x + y

def read_mesh(mesh_file):
    mesh = meshio.read(mesh_file)
    points = mesh.points[:, :2]                 # coordinates without Z because we are in 2D
    triangles = mesh.cells_dict["triangle"]
    return points, triangles, len(points), len(triangles)


def calculate_vector_and_matrix(mesh_file, f = f, c_val=1.0):
    points, triangles, num_nodes, num_triangles = read_mesh(mesh_file)

    A_global = np.zeros((num_nodes, num_nodes))
    B_global = np.zeros(num_nodes)
    
    # Assembly loop
    for e in range(num_triangles):
        nodes_tri = triangles[e]
        
        # Coordinates of the triangle vertices
        x1, y1 = points[nodes_tri[0]]
        x2, y2 = points[nodes_tri[1]]
        x3, y3 = points[nodes_tri[2]]
        
        # Calculate Jacobian matrix B_K
        B_K = np.array([
            [x2 - x1, x3 - x1],
            [y2 - y1, y3 - y1]
        ])
        
        # Determinant and Area
        det_B_K = np.linalg.det(B_K)
        area = abs(det_B_K) / 2.0
        
        # Inverse transpose of B_K for gradient transformation
        inv_B_K_T = np.linalg.inv(B_K).T
        
        # Real gradients of shape functions
        grad_N_real = np.zeros((3, 2))
        for i in range(3):
            grad_N_real[i] = inv_B_K_T @ grad_N_ref[i]        


        # Matrix A loop
        A_local = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                W_ij = np.dot(grad_N_real[i], grad_N_real[j]) * area

                if i == j:
                    M_ij = area / 6.0       # diagonal
                else:
                    M_ij = area / 12.0      # out of diagonal
                
                # A = W + c*M
                A_local[i, j] = W_ij + c_val * M_ij

        

        # Vector B loop
        B_local = np.zeros(3) 
        
        # Calculate the value of f at the centroid of the triangle
        x_c = (x1 + x2 + x3) / 3.0
        y_c = (y1 + y2 + y3) / 3.0
        f_val = f(x_c, y_c)

        for i in range(3):
            B_local[i] = f_val * (area / 3.0)



        # Final assembly into global A and B
        for i in range(3):
            global_i = nodes_tri[i]
            B_global[global_i] += B_local[i]

            for j in range(3):
                global_j = nodes_tri[j]
                A_global[global_i, global_j] += A_local[i, j]
                
    return A_global, B_global, points

if __name__ == '__main__':
    A, puntos = setup_fem_2d("mesh.msh", c_val=1.0)
    print(f"Dimensiones de la matriz global A: {A.shape}")