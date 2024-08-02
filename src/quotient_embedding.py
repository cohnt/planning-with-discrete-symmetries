import numpy as np
import functools
from dataclasses import dataclass

def tensordot(vecs):
	return functools.reduce(lambda u, v: np.tensordot(u, v, axes=0), vecs)

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

	def E_alpha_u(self, R):
		return np.hstack([
			tensordot([R @ u_i] * alpha_i).flatten()
			for u_i, alpha_i in zip(self.u, self.alpha)
		]).flatten()

	def E_u_alpha_S(self, R):
		return np.sum([
			self.E_alpha_u(R @ S_i) for S_i in self.S.matrices
		]) / self.S.order()

	def E_alpha_beta_u(self, R):
		return np.hstack([
			beta_i * tensordot([R @ u_i] * alpha_i).flatten()
			for beta_i, u_i, alpha_i in zip(self.beta, self.u, self.alpha)
		]).flatten()

	def E_alpha_beta_u_S(self, R):
		return np.sum([
			self.E_alpha_beta_u(R @ S_i) for S_i in self.S.matrices
		], axis=0) / self.S.order()

	def __call__(self, R):
		return self.E_alpha_beta_u_S(R)

if __name__ == "__main__":
	import symmetry

	# Cyclic group with two elements
	alpha = (1, 2, 2)
	u = np.eye(3)
	S = symmetry.CyclicGroupSO3(2)
	beta = (1/np.sqrt(2), 1/2, 1/2)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Should be practically zero:", np.max(np.var(out, axis=0)))

	# Octahedral group
	alpha = (4,)
	u = [np.array([1, 0, 0])]
	S = symmetry.OctahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Should be practically zero:", np.max(np.var(out, axis=0)))

	# Icosahedral group
	alpha = (10,)
	u = [np.array([1, 0, 0])]
	S = symmetry.IcosahedralGroup()
	beta = (75 / (8 * np.sqrt(95)),)
	E = Embedding(alpha, u, S, beta)

	R1 = np.eye(3)
	orbit = S.orbit(R1)
	out = [E(R) for R in orbit]
	print("Should be practically zero:", np.max(np.var(out, axis=0)))