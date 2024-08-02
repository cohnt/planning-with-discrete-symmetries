import numpy as np
import functools
import itertools
from tqdm import tqdm

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
	def __init__(self, alpha, u, S, beta=None):
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

	def compute_M_alpha(self):
		M_alpha = []
		for alpha_i in self.alpha:
			if alpha_i % 2 == 0:
				M_alpha.append(symmetrized_tensor_identity(alpha_i).flatten())
			else:
				M_alpha.append(np.zeros(3 ** alpha_i))
		return M_alpha

		# M_alpha_entries = []
		# for i in range(len(self.alpha)):
		# 	alpha_i = self.alpha[i]
		# 	if (alpha_i + 1) % 2 == 0:
		# 		mult = np.math.factorial(alpha_i+1)
		# 		M_alpha_i = np.zeros(np.power(tuple([3] * alpha_i + 1)))
		# 		M_alpha_entries.append(symmetrize(tensordot([np.eye(3)] * int((alpha_i + 1) / 2))))
		# 	else:
		# 		M_alpha_entries.append(tensordot([np.zeros(3)] * alpha_i).flatten())
		# return tensordot(M_alpha_entries).flatten()

	def E_alpha_u(self, R):
		return np.hstack([
			tensordot([R @ u_i] * alpha_i).flatten()
			for u_i, alpha_i in zip(self.u, self.alpha)
		]).flatten()

	def E_alpha_u_S(self, R):
		return np.sum([
			self.E_alpha_u(R @ S_i) for S_i in self.S.matrices
		]) / self.S.order()

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

	def __call__(self, R):
		return self.tilde_E_alpha_beta_u_S(R)

if __name__ == "__main__":
	import symmetry

	# print(symmetrized_tensor_identity(4))

	print("\nCyclic group with two elements")
	alpha = (1, 2, 2)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(2)
	beta = (1/np.sqrt(2), 1/2, 1/2)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))

	print("\nOctahedral group")
	alpha = (4,)
	u = [np.array([1, 0, 0])]
	S = symmetry.OctahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))
	print("M_alpha", E.M_alpha)

	print("\nIcosahedral group")
	alpha = (10,)
	u = [np.array([1, 0, 0])]
	S = symmetry.IcosahedralGroup()
	beta = (75 / (8 * np.sqrt(95)),)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Dim", out[0].shape)
	print("Should be practically zero:", np.max(np.var(out, axis=0)))