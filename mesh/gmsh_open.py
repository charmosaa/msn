import gmsh
import sys

gmsh.initialize()

try:
    gmsh.open("cuadrado.geo")
except:
    print("Could not find the .geo file.")
    gmsh.finalize()
    exit()


gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
gmsh.write("output_mesh.msh")

# Launch the GUI so you can inspect the result
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()