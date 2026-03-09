from utils import write_inp_file
import meshio
import numpy as np

C_VAL = 1.0

def f(x, y):
    return x*y

grad_N_ref = np.array([
    [-1.0, -1.0], 
    [ 1.0,  0.0], 
    [ 0.0,  1.0]
])

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

        gauss_points = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
        w = 1/6
        
        for xi, eta in gauss_points:
            x_real = x1 + xi*(x2 - x1) + eta*(x3 - x1)
            y_real = y1 + xi*(y2 - y1) + eta*(y3 - y1)

            f_val = f(x_real, y_real)
            N = np.array([1 - xi - eta, xi, eta])  # shape functions at the Gauss point
            
            for i in range(3):
                B_local[i] += f_val * N[i] * abs(det_B_K) * w


        # Final assembly into global A and B
        for i in range(3):
            global_i = nodes_tri[i]
            B_global[global_i] += B_local[i]

            for j in range(3):
                global_j = nodes_tri[j]
                A_global[global_i, global_j] += A_local[i, j]
                
    return A_global, B_global, points


def apply_dirichlet_bc(A, B, points, boundary_nodes, g):
    for node in boundary_nodes:
        A[node, :] = 0
        A[node, node] = 1
        B[node] = g(points[node][0], points[node][1])
    return A, B


def solve_fem_2d_dirichlet(mesh_file, g_lambdas, c_val=1.0):
    A, B, points = calculate_vector_and_matrix(mesh_file, c_val=c_val)
    mesh = meshio.read(mesh_file)

    # 1:bottom, 2:right, 3:top, 4:left
    tag_to_boundary_index = {1: 0, 2: 1, 3: 2, 4: 3}

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == "line":
            physical_tags = mesh.cell_data["gmsh:physical"][i]
            
            for j, tag in enumerate(physical_tags):
                if tag in tag_to_boundary_index:
                    boundary_idx = tag_to_boundary_index[tag]
                    node_indices = cell_block.data[j]

                    A, B = apply_dirichlet_bc(A, B, points, node_indices, g_lambdas[boundary_idx])

    u = np.linalg.solve(A, B)

    return points, u

if __name__ == '__main__':
    mesh_file = "squareMesh.msh"
    points, u = solve_fem_2d_dirichlet(mesh_file, [lambda x, y: 1, lambda x, y: 0, lambda x, y: 1, lambda x, y: y], c_val=C_VAL)
    
    # Get the triangles for the .inp file
    mesh = meshio.read(mesh_file)
    triangles = mesh.cells_dict["triangle"]
    
    # Save for ParaView
    write_inp_file("result_2d.inp", points, triangles, u)