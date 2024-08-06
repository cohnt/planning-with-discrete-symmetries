import numpy as np
import functools
import itertools

from scipy.stats import special_ortho_group
from scipy.special import binom

from pydrake.all import VPolytope, AffineSubspace

def tensordot(vecs):
	return functools.reduce(lambda u, v: np.tensordot(u, v, axes=0), vecs)

def symmetrized_tensor_identity(alpha):
	assert isinstance(alpha, int)
	assert alpha % 2 == 0
	M_alpha = np.zeros([3] * alpha)
	order = alpha
	permutations = itertools.permutations(list(range(order)))

	for i in range(M_alpha.size):
		idx_str = np.base_repr(i, 3).zfill(order)
		idx = tuple([int(foo) for foo in idx_str])
		for permutation in permutations:
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

		U, S, V = np.linalg.svd(ys - ys[:,[0]])
		dimension = np.count_nonzero(np.abs(S) > tol)
		basis = U[:,0:dimension]

		approx_center = np.mean(ys, axis=1)
		return AffineSubspace(basis, approx_center)

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

def B1NormSq(k):
	assert isinstance(k, int)
	if k % 2 == 1:
		return (k**2) / (2**(k-1))
	else:
		a = -k * (k-1) / (2**(k-2))
		b = binom(k-2, (k/2)-1)
		c = (k**2) / (2**k)
		d = 2 + binom(k, k/2)
		return (a*b) + (c*d)

def B2NormSq(k):
	assert isinstance(k, int)
	if k % 2 == 1:
		return k / (2**k)
	else:
		e = k / (2**(k+1))
		f = 2
		g = binom(k-1, k/2)
		h = binom(k-1, (k/2)-1)
		return e * (f + g + h)

def beta_C(k):
	assert isinstance(k, int)
	assert k > 2
	return (
		np.sqrt(1 - (B2NormSq(k) / B1NormSq(k))),
		1 / np.sqrt(B1NormSq(k))
	)

def beta_D(k):
	assert isinstance(k, int)
	assert k > 2
	foo = beta = beta_C(k)
	return (foo[0] / np.sqrt(2), foo[1])

def C1():
	alpha = (1, 1, 1)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(1)
	beta = tuple([1/np.sqrt(2)] * 3)
	return Embedding(alpha, u, S, beta)

def C2():
	alpha = (1, 2, 2)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(2)
	beta = (1/np.sqrt(2), 1/2, 1/2)
	return Embedding(alpha, u, S, beta)

def CN(n):
	alpha = (1, n)
	u = [[1, 0, 0], [0, 1, 0]]
	S = symmetry.CyclicGroupSO3(n)
	beta = beta_C(n)
	return Embedding(alpha, u, S, beta)

def D2():
	alpha = (2,2,2)
	u = np.eye(3)
	S = symmetry.DihedralGroup(2)
	beta = tuple([1/2] * 3)
	return Embedding(alpha, u, S, beta)

def DN(n):
	alpha = (2,n)
	u = [[1, 0, 0], [0, 1, 0]]
	S = symmetry.DihedralGroup(3)
	beta = beta_D(n)
	return Embedding(alpha, u, S, beta)

def T():
	alpha = (3,)
	vec = np.array([0.5, 0.5, -0.5])
	u = [vec / np.linalg.norm(vec)]
	S = symmetry.TetrahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	return Embedding(alpha, u, S, beta)

def O():
	alpha = (4,)
	u = [[1, 0, 0]]
	u[0] = u[0] / np.linalg.norm(u[0])
	S = symmetry.OctahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	return Embedding(alpha, u, S, beta)

def Y():
	alpha = (10,)
	u = [[1, 0, 0]]
	S = symmetry.IcosahedralGroup()
	beta = (75 / (8 * np.sqrt(95)),)
	return Embedding(alpha, u, S, beta)

if __name__ == "__main__":
	import symmetry

	assert np.allclose(beta_C(3), (np.sqrt(5/6), np.sqrt(4/9)))
	assert np.allclose(beta_C(4), (np.sqrt(1/2), np.sqrt(1/2)))
	assert np.allclose(beta_C(6), (np.sqrt(1/12), np.sqrt(8/9)))
	assert np.allclose(beta_D(3), (np.sqrt(5/12), np.sqrt(4/9)))
	assert np.allclose(beta_D(4), (1/2, np.sqrt(1/2)))
	assert np.allclose(beta_D(6), (np.sqrt(1/24), np.sqrt(8/9)))

	# print(symmetrized_tensor_identity(4))

	print("\nCyclic group with 1 element")
	E = C1()
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be nonzero:", np.linalg.norm(out[0]))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))

	print("\nCyclic group with 2 elements")
	E = C2()
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R, True) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nCyclic group with 3 elements")
	E = CN(3)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nCyclic group with 4 elements")
	E = CN(4)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nCyclic group with 6 elements")
	E = CN(6)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 4 elements")
	E = D2()
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 6 elements")
	E = DN(3)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 8 elements")
	E = DN(4)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	print("\nDihedral group with 12 elements")
	E = DN(6)
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))


	print("\nTetrahedral group")
	E = T()
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))


	print("\nOctahedral group")
	E = O()
	R1 = special_ortho_group.rvs(3)
	orbit = E.S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	print("Should be nonzero:", np.linalg.norm(out[0]))

	# print("\nIcosahedral group")
	# E = Y()
	# R1 = np.eye(3)
	# orbit = E.S.orbit(R1)
	# out = [E(R) for R in orbit]
	# print("Dim", out[0].shape)
	# print("Should be practically zero:", np.max(np.var(out, axis=0)))
	# print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	# print("Should be nonzero:", np.linalg.norm(out[0]))