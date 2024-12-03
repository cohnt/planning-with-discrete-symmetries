import numpy as np
import src.symmetry

def rotation_distance_so2(m1, m2):
    R = m1 @ np.moveaxis(m2, -2, -1)
    cos_theta = np.trace(R, 0, -2, -1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return theta

def rotation_distance_so3(m1, m2):
    R = m1 @ np.moveaxis(m2, -2, -1)
    cos_theta = (np.trace(R, 0, -2, -1) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return theta

def theta_to_so2(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([
        [c, -s],
        [s, c]
    ])

def so2_to_theta(m):
    x, y = m[0,0], m[1,0]
    return np.arctan2(y, x)

class DistanceSq():
    def __init__(self, G, ambient_dim, symmetry_dof_start, symmetry_weight=1.0):
        self.G = G
        self.ambient_dim = ambient_dim
        self.symmetry_dof_start = symmetry_dof_start
        self.symmetry_weight = symmetry_weight

    def __call__(self, q1, q2):
        # Should return a tuple (distance_sq, q2'), where q2' is in the orbit of
        # q2, and the value the distance_sq was actually computed for
        pass

class SO2DistanceSq(DistanceSq):
    def __call__(self, q1, q2):
        assert len(q1) == self.ambient_dim
        assert len(q2) == self.ambient_dim
        m1 = theta_to_so2(q1[self.symmetry_dof_start])
        m2 = theta_to_so2(q2[self.symmetry_dof_start])
        m2_prime = self.G.orbit(m2)
        distances = rotation_distance_so2(m1, m2_prime)
        idx = np.argmin(distances)
        symmetry_distance = distances[idx]
        remaining_distance_sq = np.sum((np.delete(q1, [self.symmetry_dof_start]) - np.delete(q2, [self.symmetry_dof_start])) ** 2)
        total_distance_sq = self.symmetry_weight * symmetry_distance**2 + remaining_distance_sq
        q_out = q2.copy()
        q_out[self.symmetry_dof_start] = so2_to_theta(m2_prime[idx])
        return total_distance_sq, q_out

class Interpolate():
    def __init__(self, G, ambient_dim, symmetry_dof_start):
        self.G = G
        self.ambient_dim = ambient_dim
        self.symmetry_dof_start = symmetry_dof_start

    def __call__(self, q1, q2, t):
        # Should return a configuration q at position t along the minimizing
        # geodesic from q1 to q2, where t=0 is q1 and t=1 is q2
        pass

class SO2Interpolate(Interpolate):
    def __call__(self, q1, q2, t):
        assert 0 <= t and t <= 1
        if np.abs(q2[self.symmetry_dof_start] - q1[self.symmetry_dof_start]) <= np.pi:
            return t * q2 + (1 - t) * q1
        else:
            vec = np.zeros_like(q1, dtype=float)
            vec[self.symmetry_dof_start] = 2 * np.pi
            if q2[self.symmetry_dof_start] > q1[self.symmetry_dof_start]:
                return t * q2 + (1 - t) * (q1 + vec)
            else:
                return t * (q2 + vec) + (1 - t) * q1

class SampleUniform():
    def __init__(self, G, ambient_dim, symmetry_dof_start, limits_lower, limits_upper):
        self.G = G
        self.ambient_dim = ambient_dim
        self.symmetry_dof_start = symmetry_dof_start

        # Should contain entries for the symmetry components as well, but they
        # will be ignored.
        self.limits_lower = limits_lower
        self.limits_upper = limits_upper

    def __call__(self, n):
        # returns n uniform samples from the space, as an array of shape (n, dim)
        pass

class SO2SampleUniform(SampleUniform):
    def __init__(self, G, ambient_dim, symmetry_dof_start, limits_lower, limits_upper):
        super().__init__(G, ambient_dim, symmetry_dof_start, limits_lower, limits_upper)
        self.limits_lower[self.symmetry_dof_start] = -np.pi
        self.limits_upper[self.symmetry_dof_start] = np.pi

    def __call__(self, n):
        return np.random.uniform(low=self.limits_lower, high=self.limits_upper, size=(n, self.ambient_dim))

def UnwrapToContinuousPath2d(G, path, symmetry_idx):
    if len(path) == 0:
        return []

    new_path = [path[0][0], path[0][1]]
    for start, end in path[1:]:
        mat_old = theta_to_so2(new_path[-1][symmetry_idx])
        mat_new = theta_to_so2(start[symmetry_idx])
        
        assert G.equivalent(mat_old, mat_new)
        
        orbited = G.orbit(mat_new)
        dists = np.linalg.norm(orbited - mat_old, axis = (1, 2))
        best_new = orbited[np.argmin(dists)]

        # tf @ mat_new = best_new, and SO(2) means the inverse is the transpose
        tf = best_new @ mat_new.T
        theta_next = end[symmetry_idx]
        mat_next = theta_to_so2(theta_next)
        theta_next = so2_to_theta(tf @ mat_next)

        new_path.append(end)
        new_path[-1][symmetry_idx] = theta_next

        if new_path[-1][symmetry_idx] >= new_path[-2][symmetry_idx] + np.pi:
            new_path[-1][symmetry_idx] -= 2 * np.pi
        if new_path[-1][symmetry_idx] <= new_path[-2][symmetry_idx] - np.pi:
            new_path[-1][symmetry_idx] += 2 * np.pi

    return new_path

if __name__ == "__main__":
    G = src.symmetry.CyclicGroupSO2(3)
    D = SO2DistanceSq(G, 3, 2)
    q1 = np.array([0, 0, 0])
    q2 = np.array([0, 0, 2 * np.pi / 3])
    print(D(q1, q2))

    I = SO2Interpolate(G, 3, 2)
    print()
    print(I(q1, q2, 0))
    print(I(q1, q2, 0.5))
    print(I(q1, q2, 1))

    q3 = np.array([0, 0, 4 * np.pi / 3])
    print()
    print(I(q1, q3, 0))
    print(I(q1, q3, 0.5))
    print(I(q1, q3, 1))

    S = SO2Sample(G, 3, 2, [-1, -1, 0], [1, 1, 0])
    print()
    print(S(1))
    print(S(5))