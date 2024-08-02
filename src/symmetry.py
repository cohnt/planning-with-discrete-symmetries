import numpy as np
from enum import Enum
import itertools

from pydrake.all import (
	AngleAxis,
)

class SymmetryGroupBase():
	def __init__(self):
		self.action_dim = -1
		self.matrices = []

	def orbit(self, point):
		# point should be an orthogonal matrix of size (self.action_dim,
		# self.action_dim). This method will return a list of matrices
		# corresponding to all possible equivalent orientations.
		assert isinstance(point, np.ndarray)
		assert point.shape == (self.action_dim, self.action_dim)
		return np.stack(self.matrices) @ point

	def order(self):
		return len(self.matrices)

class SymmetryGroupSO3Base(SymmetryGroupBase):
	def __init__(self):
		super().__init__()
		self.action_dim = 3

# Constructs the cyclic group of order n, assuming the first standard basis
# vector is the axis of symmetry.
class CyclicGroupSO3(SymmetryGroupSO3Base):
	def __init__(self, n):
		super().__init__()
		self.matrices = []
		for i in range(n):
			theta = 2 * np.pi * i / n
			c, s = np.cos(theta), np.sin(theta)
			mat = np.array([
				[1, 0,  0],
				[0, c, -s],
				[0, s,  c]
			])
			self.matrices.append(mat)
		self.matrices = np.array(self.matrices)

# Constructs the dihedral group of order 2n, assuming the first standard basis
# vector is the axis of symmetry, and a rotation of pi radians about the second
# standard basis vector is leaves the object unchanged.
class DihedralGroup(SymmetryGroupSO3Base):
	def __init__(self, n):
		super().__init__()
		self.matrices = []
		for i in range(n):
			theta = 2 * np.pi * i / n
			c, s = np.cos(theta), np.sin(theta)
			mat = np.array([
				[1, 0,  0],
				[0, c, -s],
				[0, s,  c]
			])
			self.matrices.append(mat)
			flip = np.array([
				[-1, 0,  0],
				[ 0, 1,  0],
				[ 0, 0, -1]
			])
			self.matrices.append(mat @ flip)
		self.matrices = np.array(self.matrices)

class TetrahedralGroup(SymmetryGroupSO3Base):
	def __init__(self):
		super().__init__()
		self.matrices = [np.eye(3)]
		# Vertices are listed to match the vertices used in tetrahedron.obj
		vertices = [
			np.array([-0.5, -0.5, -0.5]),
			np.array([-0.5,  0.5,  0.5]),
			np.array([ 0.5, -0.5,  0.5]),
			np.array([ 0.5,  0.5, -0.5]),
		]
		# For details, see
		# https://en.wikipedia.org/wiki/Tetrahedral_symmetry#Chiral_tetrahedral_symmetry
		for axis in vertices:
			for angle in [-2 * np.pi / 3, 2 * np.pi / 3]:
				self.matrices.append(AngleAxis(angle,
						axis / np.linalg.norm(axis)).rotation())
		edge_midpoints = [
			0.5 * (vertices[0] + vertices[1]),
			0.5 * (vertices[0] + vertices[2]),
			0.5 * (vertices[0] + vertices[3]),
		]
		for axis in edge_midpoints:
			self.matrices.append(AngleAxis(np.pi,
					axis / np.linalg.norm(axis)).rotation())
		assert len(self.matrices) == 12

class OctahedralGroup(SymmetryGroupSO3Base):
	def __init__(self):
		super().__init__()
		self.matrices = []
		permutations = itertools.permutations([0, 1, 2])
		# For details, see
		# https://en.wikipedia.org/wiki/Octahedral_symmetry#Rotation_matrices
		for a, b, c in permutations:
			for i0 in [-1, 1]:
				for i1 in [-1, 1]:
					for i2 in [-1, 1]:
						mat = np.zeros((3,3), dtype=int)
						mat[0,a] = i0
						mat[1,b] = i1
						mat[2,c] = i2
						if np.linalg.det(mat) > 0:
							self.matrices.append(mat)
		assert len(self.matrices) == 24

class IcosahedralGroup(SymmetryGroupSO3Base):
	def __init__(self):
		super().__init__()
		self.matrices = [np.eye(3)]
		# Taken from dodecahedron.obj and icosahedron.obj, but reordered so the
		# opposing vertices follow each other.
		dodecahedron_vertices = [
			np.array([-0.57735, -0.57735, 0.57735]),
			np.array([0.57735, 0.57735, -0.57735]),
			np.array([0.934172, 0.356822, 0]),
			np.array([-0.934172, -0.356822, 0]),
			np.array([0.934172, -0.356822, 0]),
			np.array([-0.934172, 0.356822, 0]),
			np.array([0, 0.934172, 0.356822]),
			np.array([0, -0.934172, -0.356822]),
			np.array([0, 0.934172, -0.356822]),
			np.array([0, -0.934172, 0.356822]),
			np.array([0.356822, 0, -0.934172]),
			np.array([-0.356822, 0, 0.934172]),
			np.array([0.356822, 0, 0.934172]),
			np.array([-0.356822, 0, -0.934172]),
			np.array([0.57735, 0.57735, 0.57735]),
			np.array([-0.57735, -0.57735, -0.57735]),
			np.array([-0.57735, 0.57735, -0.57735]),
			np.array([0.57735, -0.57735, 0.57735]),
			np.array([0.57735, -0.57735, -0.57735]),
			np.array([-0.57735, 0.57735, 0.57735]),
		]
		icosahedron_vertices = [
			np.array([0, -0.525731, 0.850651]),
			np.array([0, 0.525731, -0.850651]),
			np.array([0.850651, 0, 0.525731]),
			np.array([-0.850651, 0, -0.525731]),
			np.array([0.850651, 0, -0.525731]),
			np.array([-0.850651, 0, 0.525731]),
			np.array([-0.525731, 0.850651, 0]),
			np.array([0.525731, -0.850651, 0]),
			np.array([0.525731, 0.850651, 0]),
			np.array([-0.525731, -0.850651, 0]),
			np.array([0, -0.525731, -0.850651]),
			np.array([0, 0.525731, 0.850651]),
		]

		# Check that subsequent elements are approximately opposite
		# for i in range(0, len(dodecahedron_vertices), 2):
		# 	assert np.linalg.norm(dodecahedron_vertices[i] + dodecahedron_vertices[i+1]) < 1e-3
		# for i in range(0, len(icosahedron_vertices), 2):
		# 	assert np.linalg.norm(icosahedron_vertices[i] + icosahedron_vertices[i+1]) < 1e-3

		# For details, see
		# https://en.wikipedia.org/wiki/Icosahedral_symmetry#Conjugacy_classes
		angles_5 = [-4 * np.pi / 5, -2 * np.pi / 5, 2 * np.pi / 5, 4 * np.pi / 5]
		for axis in icosahedron_vertices[::2]:
			for angle in angles_5:
				self.matrices.append(AngleAxis(angle,
						axis / np.linalg.norm(axis)).rotation())
		for axis in dodecahedron_vertices[::2]:
			for angle in [-2 * np.pi / 3, 2 * np.pi / 3]:
				self.matrices.append(AngleAxis(angle,
						axis / np.linalg.norm(axis)).rotation())

		edge_dist_lower = 0.71
		edge_dist_upper = 0.72
		# The actual edge distance is 0.7136438614379024
		candidate_edges = []
		for i in range(len(dodecahedron_vertices)):
			for j in range(i+1, len(dodecahedron_vertices)):
				d = np.linalg.norm(dodecahedron_vertices[i] - dodecahedron_vertices[j])
				if d > edge_dist_lower and d < edge_dist_upper:
					candidate_edges.append((i,j))
		midpoints = [0.5 * (dodecahedron_vertices[i] + dodecahedron_vertices[j])
						for i,j in candidate_edges]
		assert len(midpoints) == 30
		midpoints_unique = []
		for midpoint in midpoints:
			keep = True
			for candidate_match in midpoints_unique:
				if np.linalg.norm(midpoint + candidate_match) < 1e-3:
					keep = False
					break
			if keep:
				midpoints_unique.append(midpoint)
		assert len(midpoints_unique) == 15
		for axis in midpoints_unique:
			self.matrices.append(AngleAxis(np.pi,
					axis / np.linalg.norm(axis)).rotation())

		assert len(self.matrices) == 60

class SymmetryGroupSO2Base(SymmetryGroupBase):
	def __init__(self):
		super().__init__()
		self.action_dim = 2

# Constructs the cyclic group of order n, assuming the first standard basis
# vector is the axis of symmetry.
class CyclicGroupSO2(SymmetryGroupSO2Base):
	def __init__(self, n):
		super().__init__()
		self.matrices = []
		for i in range(n):
			theta = 2 * np.pi * i / n
			c, s = np.cos(theta), np.sin(theta)
			mat = np.array([
				[c, -s],
				[s,  c]
			])
			self.matrices.append(mat)
		self.matrices = np.array(self.matrices)

if __name__ == "__main__":
	CyclicGroupSO2(3)
	CyclicGroupSO3(3)
	DihedralGroup(3)
	TetrahedralGroup()
	OctahedralGroup()
	IcoashedralGroup()