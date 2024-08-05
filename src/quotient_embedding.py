import numpy as np
import functools
import itertools
from scipy.stats import special_ortho_group
from tqdm import tqdm

from pydrake.all import VPolytope, AffineSubspace

def tensordot(vecs):
	return functools.reduce(lambda u, v: np.tensordot(u, v, axes=0), vecs)

def symmetrized_tensor_identity(alpha):
	assert isinstance(alpha, int)
	assert alpha % 2 == 0
	M_alpha = np.zeros([3] * alpha)
	order = alpha

	for i in tqdm(range(M_alpha.size)):
		idx_str = np.base_repr(i, 3).zfill(order)
		idx = tuple([int(foo) for foo in idx_str])
		for permutation in itertools.permutations(list(range(order))):
			bar = 1
			for j1, j2 in zip(permutation[::2], permutation[1::2]):
				bar *= idx[j1] == idx[j2]
			M_alpha[idx] += bar
		M_alpha[idx] /= np.math.factorial(order)
	return M_alpha

class Embedding:
	def __init__(self, alpha, u, S, beta=None, dimension_upper_bound=100):
		self.n = len(alpha)
		self.alpha = alpha
		self.u = u
		self.S = S
		self.beta = np.ones(self.n) if beta is None else beta

		assert len(self.alpha) == self.n
		assert len(self.u) == self.n
		for alpha_i in self.alpha:
			assert isinstance(alpha_i, int)
		for u_i in self.u:
			assert len(u_i) == 3
		assert len(self.beta) == self.n

		self.M_alpha = self.compute_M_alpha()
		self.scaled_M_alpha = np.hstack([M_alpha_i / (alpha_i + 1) for M_alpha_i, alpha_i in zip(self.M_alpha, self.alpha)])

		self.dimension_upper_bound = dimension_upper_bound

		self.affine_hull = self.compute_affine_hull()
		print("Affine hull dimension", self.affine_hull.AffineDimension())
		# print("Affine hull condtion number", np.linalg.cond(self.affine_hull.basis()))

	def compute_M_alpha(self):
		M_alpha = []
		for alpha_i in self.alpha:
			if alpha_i % 2 == 0:
				M_alpha.append(symmetrized_tensor_identity(alpha_i).flatten())
			else:
				M_alpha.append(np.zeros(3 ** alpha_i))
		return M_alpha

	def compute_affine_hull(self):
		np.random.seed(0)
		tol = 1e-12

		# Generate a number of random SO(3) matrices, pass them through the
		# mapping, and compute the affine hull.
		xs = special_ortho_group.rvs(3, self.dimension_upper_bound)
		ys = np.array([self(x, project=False) for x in xs]).T
		vpoly = VPolytope(ys)
		raw_ah = AffineSubspace(vpoly, tol)
		approx_center = np.mean(ys, axis=1)
		return AffineSubspace(raw_ah.basis(), approx_center)
		# print(np.linalg.norm(np.zeros(ys.shape[0]) - raw_ah.Projection(np.zeros(ys.shape[0]))[1]))
		# return AffineSubspace(raw_ah.basis(), np.zeros(ys.shape[0]))

	def E_alpha_u(self, R):
		return np.hstack([
			tensordot([R @ u_i] * alpha_i).flatten()
			for u_i, alpha_i in zip(self.u, self.alpha)
		]).flatten()

	def E_alpha_u_S(self, R):
		return np.sum([
			self.E_alpha_u(R @ S_i) for S_i in self.S.matrices
		], axis=0) / self.S.order()

	def E_alpha_beta_u(self, R):
		return np.hstack([
			beta_i * tensordot([R @ u_i] * alpha_i).flatten()
			for beta_i, u_i, alpha_i in zip(self.beta, self.u, self.alpha)
		])

	def E_alpha_beta_u_S(self, R):
		return np.sum([
			self.E_alpha_beta_u(R @ S_i) for S_i in self.S.matrices
		], axis=0) / self.S.order()

	def tilde_E_alpha_u_S(self, R):
		return self.E_alpha_u_S(R) - self.scaled_M_alpha

	def tilde_E_alpha_beta_u_S(self, R):
		return self.E_alpha_beta_u_S(R) - self.scaled_M_alpha

	def __call__(self, R, project=True):
		# val = self.E_alpha_u_S(R)
		# val = self.E_alpha_beta_u_S(R)
		# val = self.tilde_E_alpha_u_S(R)
		val = self.tilde_E_alpha_beta_u_S(R)
		if project:
			return self.affine_hull.ToLocalCoordinates(val).flatten()
		else:
			return val

if __name__ == "__main__":
	import symmetry

	# print(symmetrized_tensor_identity(4))

	e1 = [1, 0, 0]
	e2 = [0, 1, 0]
	e3 = [0, 0, 1]

	print("\nCyclic group with 1 element")
	alpha = (1, 1, 1)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(1)
	beta = tuple([1/np.sqrt(2)] * 3)
	E = Embedding(alpha, u, S, beta)
	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be nonzero:", np.linalg.norm(out[0]))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))

	print("\nCyclic group with 2 elements")
	alpha = (1, 2, 2)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(2)
	beta = (1/np.sqrt(2), 1/2, 1/2)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R, True) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nCyclic group with 3 elements")
	alpha = (1, 3)
	u = [e1, e2]
	S = symmetry.CyclicGroupSO3(3)
	beta = (np.sqrt(5/6), np.sqrt(4)/3)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nCyclic group with 4 elements")
	alpha = (1, 4)
	u = [e1, e2]
	S = symmetry.CyclicGroupSO3(4)
	beta = (np.sqrt(1/2), np.sqrt(1/2))
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	# print("\nCyclic group with 6 elements")
	# alpha = (1, 6)
	# u = [e1, e2]
	# S = symmetry.CyclicGroupSO3(6)
	# beta = (np.sqrt(1/12), np.sqrt(8/9))
	# E = Embedding(alpha, u, S, beta)

	# R1 = special_ortho_group.rvs(3)
	# orbit = S.orbit(R1)
	# out = [E(R) for R in orbit]
	# print("Dim", out[0].shape)
	# print("Should be practically zero:", np.max(np.var(out, axis=0)))
	# print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	# print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 4 elements")
	alpha = (2,2,2)
	u = [e1, e2, e3]
	S = symmetry.DihedralGroup(2)
	beta = tuple([1/2] * 3)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 6 elements")
	alpha = (2,3)
	u = [e1, e2]
	S = symmetry.DihedralGroup(3)
	beta = [np.sqrt(5/12), np.sqrt(4/9)]
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 8 elements")
	alpha = (2,4)
	u = [e1, e2]
	S = symmetry.DihedralGroup(4)
	beta = [1/2, np.sqrt(1/2)]
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 12 elements")
	alpha = (2,3)
	u = [np.array([1, 0, 0]), np.array([0, 1, 0])]
	S = symmetry.DihedralGroup(6)
	beta = (np.sqrt(5/12), np.sqrt(4)/3)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))


	print("\nTetrahedral group")
	alpha = (3,)
	vec = np.array([0.5, 0.5, -0.5])
	u = [vec / np.linalg.norm(vec)]
	S = symmetry.TetrahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))


	print("\nOctahedral group")
	alpha = (4,)
	u = [np.array([1, 0, 0])]
	u[0] = u[0] / np.linalg.norm(u[0])
	S = symmetry.OctahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	E = Embedding(alpha, u, S, beta)

	R1 = special_ortho_group.rvs(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	# print("\nIcosahedral group")
	# alpha = (10,)
	# u = [np.array([1, 0, 0])]
	# S = symmetry.IcosahedralGroup()
	# beta = (75 / (8 * np.sqrt(95)),)
	# E = Embedding(alpha, u, S, beta)

	# R1 = np.eye(3)
	# orbit = S.orbit(R1)
	# out = [E(R) for R in orbit]
	# print("Dim", out[0].shape)
	# print("Should be practically zero:", np.max(np.var(out, axis=0)))
	# print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	# print("Should be nonzero:", np.linalg.norm(out[0]))