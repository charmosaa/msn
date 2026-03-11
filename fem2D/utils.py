def write_inp_file(filename, points, triangles, u):
    num_nodes = len(points)
    num_elements = len(triangles)
    
    with open(filename, 'w') as f:
        f.write(f"{num_nodes} {num_elements} 1 0 0\n")

        for i, pt in enumerate(points):
            f.write(f"{i+1} {pt[0]} {pt[1]} 0\n")
            
        for i, tri in enumerate(triangles):
            n1, n2, n3 = tri + 1 
            f.write(f"{i+1} 1 tri {n1} {n2} {n3}\n")

        f.write("1 1\n")
        f.write("u, solution\n")

        for i, val in enumerate(u):
            f.write(f"{i+1} {val}\n")

    print(f"File '{filename}' successfully saved for ParaView.")