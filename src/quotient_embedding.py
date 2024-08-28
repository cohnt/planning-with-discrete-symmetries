import numpy as np
import jax
import jax.numpy as jnp
import functools
import itertools
import time
from tqdm import tqdm

from scipy.stats import special_ortho_group
from scipy.special import binom

from pydrake.all import VPolytope, AffineSubspace

def tensordot(vecs):
	return functools.reduce(lambda u, v: jnp.tensordot(u, v, axes=0), vecs)

def symmetrized_tensor_identity(alpha):
	assert isinstance(alpha, int)
	assert alpha % 2 == 0
	M_alpha = jnp.zeros([3] * alpha)
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
		self.beta = jnp.ones(self.n) if beta is None else beta

		assert len(self.alpha) == self.n
		assert len(self.u) == self.n
		for alpha_i in self.alpha:
			assert isinstance(alpha_i, int)
		for u_i in self.u:
			assert isinstance(u_i, jnp.ndarray)
			assert len(u_i) == 3
		assert len(self.beta) == self.n

		self.M_alpha = None
		self.scaled_M_alpha = None
		self.tilde_M_alpha = None

		self.dimension_upper_bound = dimension_upper_bound

		self.affine_hull = self.compute_affine_hull()
		# print("Affine hull dimension", self.affine_hull.AffineDimension())
		# print("Affine hull condtion number", np.linalg.cond(self.affine_hull.basis()))

	def compute_M_alpha(self):
		if self.M_alpha is not None:
			return
		
		self.M_alpha = []
		for alpha_i in self.alpha:
			if alpha_i % 2 == 0:
				self.M_alpha.append(symmetrized_tensor_identity(alpha_i).flatten())
			else:
				self.M_alpha.append(jnp.zeros(3 ** alpha_i))

		self.scaled_M_alpha = jnp.hstack([M_alpha_i / (alpha_i + 1) for M_alpha_i, alpha_i in zip(self.M_alpha, self.alpha)])
		self.tilde_M_alpha = jnp.hstack([M_alpha_i * beta_i / (alpha_i + 1) for M_alpha_i, beta_i, alpha_i in zip(self.M_alpha, self.beta, self.alpha)])

	def compute_affine_hull(self):
		tol = 1e-12

		# Generate a number of random SO(3) matrices, pass them through the
		# mapping, and compute the affine hull.
		xs = special_ortho_group.rvs(3, self.dimension_upper_bound)
		ys = np.array([self(x, project=False) for x in xs]).T

		# Calculate basis with SVD. Center the data first.
		U, S, V = np.linalg.svd(ys - ys[:,[0]])
		dimension = np.count_nonzero(np.abs(S) > tol)
		basis = U[:,0:dimension]

		# Calculate the center per https://stackoverflow.com/a/72231029/9796174
		tmp_ah = AffineSubspace(basis, ys[:,0])
		local = tmp_ah.ToLocalCoordinates(ys)
		local = local[:,:dimension+1].T
		a = np.concatenate((local, np.ones((dimension+1, 1))), axis=1)
		b = (local**2).sum(axis=1)
		x = np.linalg.solve(a, b)
		center_local = x[:-1] / 2

		return AffineSubspace(basis, tmp_ah.ToGlobalCoordinates(center_local))

	def E_alphai_ui(self, R, alpha_i, u_i):
		#
		return tensordot([R @ u_i] * alpha_i).flatten()

	def E_alphai_ui_S(self, R, alpha_i, u_i):
		return jnp.sum(jnp.array([
			self.E_alphai_ui(R @ S_j, alpha_i, u_i) for S_j in self.S.matrices
		]), axis=0) / self.S.order()

	def E_alpha_u(self, R):
		return jnp.hstack([
			self.E_alphai_ui(R, alpha_i, u_i)
			for u_i, alpha_i in zip(self.u, self.alpha)
		]).flatten()

	def E_alpha_u_S(self, R):
		return jnp.sum(jnp.array([
			self.E_alpha_u(R @ S_i) for S_i in self.S.matrices
		]), axis=0) / self.S.order()

	def E_alphai_betai_ui(self, R, alpha_i, beta_i, u_i):
		#
		return beta_i * tensordot([R @ u_i] * alpha_i).flatten()

	def E_alphai_betai_ui_S(self, R, alpha_i, beta_i, u_i):
		return jnp.sum(jnp.array([
			self.E_alphai_betai_ui(R @ S_j, alpha_i, beta_i, u_i) for S_j in self.S.matrices
		]), axis=0) / self.S.order()

	def E_alpha_beta_u(self, R):
		return jnp.hstack([
			self.E_alphai_betai_ui(R, alpha_i, beta_i, u_i)
			for beta_i, u_i, alpha_i in zip(self.beta, self.u, self.alpha)
		])

	def E_alpha_beta_u_S(self, R):
		return jnp.sum(jnp.array([
			self.E_alpha_beta_u(R @ S_i) for S_i in self.S.matrices
		]), axis=0) / self.S.order()

	def tilde_E_alpha_u_S(self, R):
		self.compute_M_alpha()
		return self.E_alpha_u_S(R) - self.scaled_M_alpha

	def tilde_E_alpha_beta_u_S(self, R):
		self.compute_M_alpha()
		# TODO: Maybe adjust per response from paper authors
		return self.E_alpha_beta_u_S(R) - self.tilde_M_alpha

	def so3_action(self, R, O):
		return R @ O

	def embedding_flat_to_tensor(self, v):
		assert len(v) == jnp.sum(3 ** jnp.array(self.alpha))
		new_v = []
		i = 0
		for alpha_i in self.alpha:
			count = 3 ** alpha_i
			new_v.append(v[i:i+count])
			i += count
		return new_v

	def embedding_action_i(self, R, v, i):
		alpha_i = self.alpha[i]
		assert len(v) == 3 ** alpha_i

		index = (list(range(1, 2 * alpha_i, 2)), list(range(alpha_i)))

		return jnp.tensordot(tensordot([R] * alpha_i), v.reshape(tuple([3] * alpha_i)), index).flatten()

	def embedding_action(self, R, v):
		v = self.embedding_flat_to_tensor(v)

		indices = [
			(list(range(1, 2 * alpha_i, 2)), list(range(alpha_i)))
			for alpha_i in self.alpha
		]

		return jnp.hstack([
			jnp.tensordot(tensordot([R] * alpha_i), v_i.reshape(tuple([3] * alpha_i)), index).flatten()
			for v_i, alpha_i, index in zip(v, self.alpha, indices)
		])

	def ToLocalCoordinates(self, v):
		#
		return self.affine_hull.ToLocalCoordinates(v).flatten()

	def ToGlobalCoordinates(self, v):
		#
		return self.affine_hull.ToGlobalCoordinates(v).flatten()

	def J_functional(self, R, T, isometric=True):
		T = self.embedding_flat_to_tensor(T)

		if isometric:
			return jnp.sum(jnp.array([
				jnp.inner(self.E_alphai_betai_ui_S(R, alpha_i, beta_i, u_i), T_i)
				for alpha_i, beta_i, u_i, T_i in zip(self.alpha, self.beta, self.u, T)
			])) / self.S.order()
		else:
			return jnp.sum(jnp.array([
				jnp.inner(self.E_alphai_ui_S(R, alpha_i, u_i), T_i)
				for alpha_i, u_i, T_i in zip(self.alpha, self.u, T)
			]))

	def J_directional_derivative(self, R, T, s, isometric=True):
		# Directional derivative of J at R in direction sR
		assert jnp.linalg.norm(s + s.T) < 1e-12
		
		T = self.embedding_flat_to_tensor(T)

		if isometric:
			return jnp.sum(jnp.array([
				jnp.sum(jnp.array([
					alpha_i * jnp.inner(
						jnp.tensordot(s, (
								self.embedding_action_i(R @ O, self.E_alphai_betai_ui(jnp.eye(3), alpha_i, beta_i, u_i), i)
							).reshape([3] * alpha_i),
							1
						).flatten(),
						T_i
					)
					for i, (alpha_i, beta_i, u_i, T_i) in enumerate(zip(self.alpha, self.beta, self.u, T))
				])) for O in self.S.matrices
			])) / self.S.order()
		else:
			return jnp.sum(jnp.array([
				jnp.sum(jnp.array([
					alpha_i * jnp.inner(
						jnp.tensordot(s, (
								self.embedding_action_i(R @ O, self.E_alphai_ui(jnp.eye(3), alpha_i, beta_i, u_i), i)
							).reshape([3] * alpha_i),
							1
						).flatten(),
						T_i
					)
					for i, (alpha_i, u_i, T_i) in enumerate(zip(self.alpha, self.u, T))
				])) for O in self.S.matrices
			])) / self.S.order()

	def J_gradient(self, R, T, isometric=True):
		s1, s2, s3 = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
		s1[2,1] = s2[2,0] = s3[1,0] = 1
		s1[1,2] = s2[0,2] = s3[0,1] = -1

		d1 = self.J_directional_derivative(R, T, s1, isometric)
		d2 = self.J_directional_derivative(R, T, s2, isometric)
		d3 = self.J_directional_derivative(R, T, s3, isometric)

		return d1 * (s1 @ R) + d2 * (s2 @ R) + d3 * (s3 @ R)

	def project_pymanopt(self, T, isometric=True, R_secret=jnp.full((3,3), jnp.inf)):
		import pymanopt
		import pymanopt.manifolds
		import pymanopt.optimizers

		Rs = special_ortho_group.rvs(3, 2 * self.S.order())
		Js = jnp.array([self.J_functional(R, T, isometric) for R in Rs])
		R = Rs[jnp.argmax(Js)]

		# print("Highest:", jnp.max(Js), "\tActual:", self.J_functional(R_secret, T, isometric))

		manifold = pymanopt.manifolds.SpecialOrthogonalGroup(n=3, k=1, retraction="polar")
		@pymanopt.function.jax(manifold)
		def cost(point):
			return -self.J_functional(point, T, isometric)

		problem = pymanopt.Problem(manifold, cost)
		verbosity = 0
		optimizer = pymanopt.optimizers.SteepestDescent(verbosity=verbosity)

		result = optimizer.run(problem, initial_point=R)

		return result.point

	def project_embedding(self, T, isometric=True, step_size=1e-2, convergence_tol=1e-6, max_iters=int(1e3), R_secret=jnp.full((3,3), jnp.inf)):
		Rs = special_ortho_group.rvs(3, 2 * self.S.order())
		Js = [self.J_functional(R, T, isometric) for R in Rs]
		R = Rs[np.argmax(Js)]

		# print("Highest:", np.max(Js), "\tActual:", self.J_functional(R_secret, T, isometric))

		global vals

		losses = []

		for i in tqdm(range(max_iters)):
		# for i in range(max_iters):
			J_old = self.J_functional(R, T, isometric)

			dR = self.J_gradient(R, T, isometric)
			dR /= 0.5

			J_new = -jnp.inf
			while J_new - J_old < -convergence_tol:
				dR *= 0.5
				U, _, VH = jnp.linalg.svd(R + dR * step_size)
				R_new = U @ VH

				diff = jnp.linalg.norm(R_new - R)

				J_new = self.J_functional(R_new, T, isometric)

				# break
				# print(np.linalg.norm(dR))
			# print(J_new)

			R = R_new
			
			vals.append(J_new - J_old)
			losses.append(J_old)

			# if jnp.abs(J_new - J_old) < convergence_tol:
			# 	break

			if diff < convergence_tol:
				losses.append(J_new)
				break

		import matplotlib.pyplot as plt
		plt.plot(losses)
		plt.show()

		return np.array(R)

	def __call__(self, R, isometric=True, centered=False, project=True):
		if not isometric and not centered:
			val = self.E_alpha_u_S(R)
		elif isometric and not centered:
			val = self.E_alpha_beta_u_S(R)
		elif not isometric and centered:
			val = self.tilde_E_alpha_u_S(R)
		else:
			val = self.tilde_E_alpha_beta_u_S(R)

		if project:
			return self.ToLocalCoordinates(val).flatten()
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
	u = jnp.eye(3)
	S = symmetry.CyclicGroupSO3(1)
	beta = tuple([1/np.sqrt(2)] * 3)
	return Embedding(alpha, u, S, beta)

def C2():
	alpha = (1, 2, 2)
	u = jnp.eye(3)
	S = symmetry.CyclicGroupSO3(2)
	beta = (1/np.sqrt(2), 1/2, 1/2)
	return Embedding(alpha, u, S, beta)

def CN(n):
	alpha = (1, n)
	u = jnp.array([[1, 0, 0], [0, 1, 0]])
	S = symmetry.CyclicGroupSO3(n)
	beta = beta_C(n)
	return Embedding(alpha, u, S, beta)

def D2():
	alpha = (2,2,2)
	u = jnp.eye(3)
	S = symmetry.DihedralGroup(2)
	beta = tuple([1/2] * 3)
	return Embedding(alpha, u, S, beta)

def DN(n):
	alpha = (2,n)
	u = jnp.array([[1, 0, 0], [0, 1, 0]])
	S = symmetry.DihedralGroup(n)
	beta = beta_D(n)
	return Embedding(alpha, u, S, beta)

def T():
	alpha = (3,)
	vec = jnp.array([0.5, 0.5, -0.5])
	u = [vec / jnp.linalg.norm(vec)]
	S = symmetry.TetrahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	return Embedding(alpha, u, S, beta)

def O():
	alpha = (4,)
	u = [jnp.array([1, 0, 0])]
	u[0] = u[0] / jnp.linalg.norm(u[0])
	S = symmetry.OctahedralGroup()
	beta = (3 / (2 * np.sqrt(2)),)
	return Embedding(alpha, u, S, beta)

def Y():
	alpha = (10,)
	u = jnp.array([[1, 0, 0]])
	S = symmetry.IcosahedralGroup()
	beta = (75 / (8 * np.sqrt(95)),)
	return Embedding(alpha, u, S, beta)

if __name__ == "__main__":
	import symmetry

	vals = []

	# print(symmetrized_tensor_identity(4))

	assert np.allclose(beta_C(3), (np.sqrt(5/6), np.sqrt(4/9)))
	assert np.allclose(beta_C(4), (np.sqrt(1/2), np.sqrt(1/2)))
	assert np.allclose(beta_C(6), (np.sqrt(1/12), np.sqrt(8/9)))
	assert np.allclose(beta_D(3), (np.sqrt(5/12), np.sqrt(4/9)))
	assert np.allclose(beta_D(4), (1/2, np.sqrt(1/2)))
	assert np.allclose(beta_D(6), (np.sqrt(1/24), np.sqrt(8/9)))

	# for E in [C1(), C2(), CN(3), CN(4), CN(5), CN(6), D2(), DN(3), DN(4), DN(5), DN(6), T(), O(), Y()]:
	for E in [C1(), C2(), CN(3), CN(4), CN(5), CN(6), D2(), DN(3), DN(4), DN(5), DN(6), T(), O()]:
	# for E in [C2()]:
		print("Group order", E.S.order(), "\tEmbedding Output Ambient Dimension", np.sum([3 ** np.array(E.alpha)]))
		# # Check equivariance
		# R, S = special_ortho_group.rvs(3, 2)
		# v1 = E.E_alpha_u_S(E.so3_action(R, S))
		# v2 = E.embedding_action(R, E.E_alpha_u_S(S))
		# v1 = E(E.so3_action(R, S), project=False)
		# v2 = E.embedding_action(R, E(S, project=False))
		# v1 = E.ToLocalCoordinates(E(E.so3_action(R, S), project=False))
		# v2 = E.ToLocalCoordinates(E.embedding_action(R, E.ToGlobalCoordinates(E(S))))
		# print("Should be near zero", np.linalg.norm(v1 - v2))

		# # Check the J functional
		# R, S = special_ortho_group.rvs(3, 2)
		# T1 = E.E_alpha_u_S(R)
		# T2 = E.E_alpha_beta_u_S(R)
		# print("Should be positive", E.J_functional(R, T1, isometric=False) - E.J_functional(S, T1, isometric=False))
		# print("Should be positive", E.J_functional(R, T2, isometric=True) - E.J_functional(S, T2, isometric=True))
		# S = E.S.orbit(R)[np.random.choice(E.S.order())]
		# print("Should be zero", E.J_functional(R, T1, isometric=False) - E.J_functional(S, T1, isometric=False))
		# print("Should be zero", E.J_functional(R, T2, isometric=True) - E.J_functional(S, T2, isometric=True))

		# # Check directional derivative of J functional
		# R = special_ortho_group.rvs(3)
		# T1 = E.E_alpha_u_S(R)
		# T2 = E.E_alpha_beta_u_S(R)
		# a, b, c = np.random.uniform([-1]*3, [1]*3)
		# s1, s2, s3 = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
		# s1[2,1] = s2[2,0] = s3[1,0] = 1
		# s1[1,2] = s2[0,2] = s3[0,1] = -1
		# s = a * s1 + b * s2 + c * s3
		# print("Should be near zero", E.J_directional_derivative(R, T1, s, isometric=False))
		# print("Should be near zero", E.J_directional_derivative(R, T2, s, isometric=True))
		# S = special_ortho_group.rvs(3)
		# T3 = E.E_alpha_u_S(S)
		# T4 = E.E_alpha_beta_u_S(S)
		# print("Should be nonzero", E.J_directional_derivative(R, T3, s, isometric=False))
		# print("Should be nonzero", E.J_directional_derivative(R, T4, s, isometric=True))

		# # Test the gradient of the J functional
		# R = special_ortho_group.rvs(3)
		# a, b, c = np.random.uniform([-0.1]*3, [0.1]*3)
		# s1, s2, s3 = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
		# s1[2,1] = s2[2,0] = s3[1,0] = 1
		# s1[1,2] = s2[0,2] = s3[0,1] = -1
		# s1 = s1 @ R
		# s2 = s2 @ R
		# s3 = s3 @ R
		# s = a * s1 + b * s2 + c * s3

		# U, _, VH = np.linalg.svd(R + s)
		# S = U @ VH
		# # print("Should be positive", E.J_functional(R, E.E_alpha_u_S(R)) - E.J_functional(R, E.E_alpha_u_S(S)))

		# grad = E.J_gradient(S, E.E_alpha_u_S(R))
		# S_new = S - (grad @ S * 0.1)
		# # U, _, VH = np.linalg.svd(S_new)
		# # S_new = U @ VH
		# print("Should be positive", E.J_functional(S_new, E.E_alpha_u_S(R)) - E.J_functional(S, E.E_alpha_u_S(R)))

		# grad_isom = E.J_gradient(S, E.E_alpha_beta_u_S(R), isometric=True)
		# S_new = S - (grad_isom @ S * 0.01)
		# # U, _, VH = np.linalg.svd(S_new)
		# # S_new = U @ VH
		# print("Should be positive", E.J_functional(S_new, E.E_alpha_beta_u_S(R), isometric=True) - E.J_functional(S, E.E_alpha_beta_u_S(R), isometric=True))

		# Check projection
		R = special_ortho_group.rvs(3)
		out = E(R, isometric=True, centered=False, project=False)

		# print(R)
		# print(E.project_embedding(out, isometric=False))
		# print("Should be nearly zero", np.linalg.norm(R - E.project_embedding(out, isometric=False)))
		# proj = E.project_embedding(out, isometric=False, step_size=1e-2, convergence_tol=1e-8, max_iters=int(1e5))
		# proj = E.project_embedding(out, isometric=True, R_secret=R)
		# print("Should be true", E.S.equivalent(R, proj, tol=1e-2))
		proj = E.project_pymanopt(out, isometric=True, R_secret=R)
		print("Should be true", E.S.equivalent(R, proj, tol=1e-2))
		# print(np.min(vals))
		# success_fail.append(E.S.equivalent(R, proj))

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
	# R1 = special_ortho_group.rvs(3)
	# orbit = E.S.orbit(R1)
	# out = [E(R) for R in orbit]
	# print("Dim", out[0].shape)
	# print("Should be practically zero:", np.max(np.var(out, axis=0)))
	# print("Should be practically zero:", np.max(np.var([np.linalg.norm(E(R)) for R in special_ortho_group.rvs(3, 10)])))
	# print("Should be nonzero:", np.linalg.norm(out[0]))