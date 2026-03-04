def write_inp_file(filename, points, triangles, u):
    num_nodes = len(points)
    num_elements = len(triangles)
    
    with open(filename, 'w') as f:
        # 1. Write Header 
        # Nodes, Elements, 1 (Scalar fields), 0, 0
        f.write(f"{num_nodes} {num_elements} 1 0 0\n")
        
        # 2. Write Node Coordinates 
        for i, pt in enumerate(points):
            # Format: index x y z (z is 0 for 2D)
            f.write(f"{i+1} {pt[0]} {pt[1]} 0\n")
            
        # 3. Write Element Connectivity 
        for i, tri in enumerate(triangles):
            # Format: index material_id type node1 node2 node3
            # Note: .inp nodes are usually 1-indexed
            n1, n2, n3 = tri + 1 
            f.write(f"{i+1} 1 tri {n1} {n2} {n3}\n")
            
        # 4. Write Scalar Data Header 
        f.write("1 1\n") # 1 field, 1 component
        f.write("u, solution\n") # label
        
        # 5. Write Scalar Values per node [cite: 988-989]
        for i, val in enumerate(u):
            f.write(f"{i+1} {val}\n")

    print(f"File '{filename}' successfully saved for ParaView.")