import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog="rectangular_pyramid.py",
    description="Generate a Rectangular Pyramid .obj File")

parser.add_argument("-f", "--fname", type=str, required=True)

args = parser.parse_args()
obj_name = "rectangular_pyramid"

vertices = []
faces = []

theta = np.pi/8
thetas = np.array([theta, np.pi - theta, np.pi + theta, -theta])
for theta in thetas:
    c, s = np.cos(theta), np.sin(theta)
    vertices.append((-0.5, c, s))

vertices.append((0.5, 0, 0)) # Top center
vertices.append((-0.5, 0, 0)) # Bottom center

tc = len(vertices) - 2
bc = len(vertices) - 1

# obj files use 1-indexing
tc += 1
bc += 1

for i in range(4):
    b1 = i
    b2 = (i + 1) % 4

    # obj files use 1-indexing
    b1 += 1
    b2 += 1

    faces.append((b1, bc, b2))
    faces.append((b1, b2, tc))

with open(args.fname, "w") as f:
    f.write("g %s\n" % obj_name)
    f.write("\n")
    for v in vertices:
        f.write("v %f %f %f\n" % v)
    f.write("\n")
    for face in faces:
        f.write("f %d %d %d\n" % face)