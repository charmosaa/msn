import numpy as np

def f(x, y):
    return np.exp(-20 * ((x - 0.5)**2 + (y - 0.5)**2))


def find_element(p, nodes, elements):
    for idx, elem in enumerate(elements):
        # print(f"Checking element {idx} with nodes {elem}")
        p1, p2, p3 = nodes[elem[0] - 1], nodes[elem[1] - 1], nodes[elem[2] - 1]

        # print(f"Vertices: {p1}, {p2}, {p3}")

        v1 = p2 - p1
        v2 = p3 - p1
        v = p - p1
        
        # v = alpha*v1 + beta*v2
        A = np.stack([v1, v2], axis=1)
        try:
            coeffs = np.linalg.solve(A, v)
            alpha, beta = coeffs
            
            if (alpha >= 0) and (beta >= 0) and (alpha + beta <= 1):
                return idx, elem, alpha, beta
        except np.linalg.LinAlgError:
            continue
    return None, None, None, None

def compute_shape_functions(p_target, tri_coords):
    M = np.column_stack((np.ones(3), tri_coords))
    
    rhs = np.eye(3)
    coeffs = np.linalg.solve(M, rhs)

    p_vector = np.array([1, p_target[0], p_target[1]])

    N = np.dot(p_vector, coeffs)
    
    return N

def interpolate(p, nodes, elements):
    _, element_nodes, alpha, beta = find_element(p, nodes, elements)
    if element_nodes is not None:
        coords = nodes[element_nodes[0]-1], nodes[element_nodes[1]-1], nodes[element_nodes[2]-1]
        z_values = np.array([f(pt[0], pt[1]) for pt in coords])
        
        n1 = 1 - alpha - beta
        n2 = alpha
        n3 = beta
        # # or we can do this as shown in the slides: 
        # n1, n2, n3 = compute_shape_functions(p, np.array(coords))

        z_interpolated = z_values[0]*n1 + z_values[1]*n2 + z_values[2]*n3
        
        return z_interpolated
    else:
        raise ValueError("Point is outside the triangulation.")

if __name__ == '__main__':   
    nodes = np.loadtxt('nodos.dat')
    elements = np.loadtxt('triangulos.dat', dtype=int) 

    p_target = np.array([0.1, 0.5])

    print(f"Interpolated value: {interpolate(p_target, nodes, elements)}")
    print(f"Real value: {f(p_target[0], p_target[1])}")