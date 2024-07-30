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
			theta = 2 * np.pi * i / self.order
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
			theta = 2 * np.pi * i / self.order
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
		self.matrices = []
		# Vertices are listed to match the vertices used in tetrahedron.obj
		vertices = [
			np.array([-0.5, -0.5, -0.5]),
			np.array([-0.5,  0.5,  0.5]),
			np.array([ 0.5, -0.5,  0.5]),
			np.array([ 0.5,  0.5, -0.5])
		]
		for axis in vertices:
			for theta in [-2 * np.pi / 3, 2 * np.pi / 3]:
				self.matrices.append(AngleAxis(theta,
						axis / np.linalg.norm(axis)).rotation())
		edge_midpoints = [
			0.5 * (vertices[0] + vertices[1]),
			0.5 * (vertices[0] + vertices[2]),
			0.5 * (vertices[0] + vertices[3])
		]
		for axis in edge_midpoints:
			self.matrices.append(AngleAxis(np.pi,
					axis / np.linalg.norm(axis)).rotation())

class OctahedralGroup(SymmetryGroupSO3Base):
	def __init__(self):
		super().__init__()
		self.matrices = []
		permutations = itertools.permutations([0, 1, 2])
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

class IcoashedralGroup(SymmetryGroupSO3Base):
	def __init__(self):
		super().__init__()
		raise(NotImplementedError)
		# TODO

class SymmetryGroupSO2Base(SymmetryGroupBase):
	def __init__(self):
		super().__init__()
		self.action_dim = 2

# Constructs the cyclic group of order n, assuming the first standard basis
# vector is the axis of symmetry.
class CyclicGroupSO2(SymmetryGroupSO2Base):
	def __init__(self, n):
		super().__init__()
		self.order = n
		self.matrices = []
		for i in range(n):
			theta = 2 * np.pi * i / self.order
			c, s = np.cos(theta), np.sin(theta)
			mat = np.array([
				[c, -s],
				[s,  c]
			])
			self.matrices.append(mat)
		self.matrices = np.array(self.matrices)