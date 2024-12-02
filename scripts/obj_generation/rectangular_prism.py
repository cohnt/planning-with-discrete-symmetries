import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="rectangular_prism.py",
    description="Generate a Rectangular Prism .obj File")

parser.add_argument("-f", "--fname", type=str, required=True)

args = parser.parse_args()
obj_name = "rectangular_prism"

vertices = []
faces = []

theta = np.pi/8
thetas = np.array([theta, np.pi - theta, np.pi + theta, -theta])
for theta in thetas:
    c, s = np.cos(theta), np.sin(theta)
    vertices.append((0.5, c, s))
    vertices.append((-0.5, c, s))

vertices.append((0.5, 0, 0)) # Top center
vertices.append((-0.5, 0, 0)) # Bottom center

tc = len(vertices) - 2
bc = len(vertices) - 1

# obj files use 1-indexing
tc += 1
bc += 1

for i in range(4):
    t1 = 2*i
    t2 = (2*(i+1)) % (2 * 4)
    b1 = t1 + 1
    b2 = t2 + 1

    # obj files use 1-indexing
    t1 += 1
    t2 += 1
    b1 += 1
    b2 += 1

    faces.append((t1, t2, tc))
    faces.append((b1, bc, b2))
    faces.append((t1, b1, t2))
    faces.append((t2, b1, b2))

with open(args.fname, "w") as f:
    f.write("g %s\n" % obj_name)
    f.write("\n")
    for v in vertices:
        f.write("v %f %f %f\n" % v)
    f.write("\n")
    for face in faces:
        f.write("f %d %d %d\n" % face)