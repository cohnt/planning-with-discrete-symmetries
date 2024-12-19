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

    def pairwise(self, q1s, q2s=None):
        # If q2s is None, compute pairwise distances within q1s
        # Should return a pair, whose first entry is the pairwise distances, in
        # the form of a matrix of shape (len(q1s), len(q2s)), and the second
        # entry is the closest elements, as a matrix of shape
        # (len(q1s), len(q2s), dim)
        pass

    def path_length(self, path):
        # Path should already be unwrapped!
        segment_lengths = [np.sqrt(self(path[i], path[i-1])[0]) for i in range(1, len(path))]
        return np.sum(segment_lengths)

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
        q_out[self.symmetry_dof_start] = WrapTheta2d(self.G, q1[self.symmetry_dof_start], so2_to_theta(m2_prime[idx]))
        return total_distance_sq, q_out

    def pairwise(self, q1s, q2s=None):
        if q2s is None:
            q2s = q1s

        np_q1s = np.asarray(q1s)
        np_q2s = np.asarray(q2s)

        # Euclidean components
        remaining_q1s = np.delete(np_q1s, self.symmetry_dof_start, axis=1)
        remaining_q2s = np.delete(np_q2s, self.symmetry_dof_start, axis=1)
        remaining_dists_squared = (
            np.sum(remaining_q1s**2, axis=1, keepdims=True)  # Shape (n, 1)
            + np.sum(remaining_q2s**2, axis=1)              # Shape (m,)
            - 2 * np.dot(remaining_q1s, remaining_q2s.T)    # Shape (n, m)
        )

        # Symmetric components
        thetas1 = np_q1s[:, self.symmetry_dof_start]
        thetas2 = np_q2s[:, self.symmetry_dof_start]
        mats1 = theta_to_so2(thetas1).T
        mats2 = theta_to_so2(thetas2).T

        orbits = np.array([self.G.orbit(mat2.T) for mat2 in mats2])  # Shape (m, orbit_len, 2, 2)
        orbited_mats2 = orbits.reshape(-1, 2, 2)  # Flatten orbits for pairwise computation

        # Compute pairwise distances.
        products = np.einsum('ijk,mkl->imjl', mats1, orbited_mats2)  # Shape (n, m * orbit_len, 2, 2)
        traces = np.trace(products, axis1=-2, axis2=-1)  # Shape (n, m * orbit_len)
        distances = np.arccos(np.clip(traces / 2, -1.0, 1.0))  # Shape (n, m * orbit_len)

        # Reshape distances to group by orbits
        distances = distances.reshape(len(mats1), len(mats2), -1)  # Shape (n, m, orbit_len)

        # Compute minimum distance
        min_distances = np.min(distances, axis=2)  # Shape (n, m)
        min_indices = np.argmin(distances, axis=2)  # Shape (n, m)

        nearest_mats = np.zeros((len(q1s), len(q2s), 2, 2))
        nearest_thetas = np.zeros((len(q1s), len(q2s)))

        # Assuming orbits is of shape (len(q2s), N, 2, 2), and min_indices is of shape (len(q1s), len(q2s))
        i_indices = np.arange(len(q1s))[:, None]  # Shape (len(q1s), 1)
        j_indices = np.arange(len(q2s))  # Shape (len(q2s),)

        # Use these indices to select from orbits
        nearest_mats = orbits[j_indices[None, :], min_indices]  # Shape (len(q1s), len(q2s), 2, 2)

        # Compute theta for each nearest matrix
        nearest_thetas = np.arctan2(nearest_mats[:, :, 1, 0], nearest_mats[:, :, 0, 0])  # Shape (n, m)
        nearest_thetas_old = nearest_thetas.copy()

        # Wrap the thetas around if necessary
        nearest_thetas = WrapTheta2dVectorized(self.G, q1s[:,self.symmetry_dof_start], nearest_thetas_old)

        # Assign theta values correctly
        nearest_entries = np.tile(q2s, (len(q1s), 1, 1))
        nearest_entries[:, :, self.symmetry_dof_start] = nearest_thetas

        return self.symmetry_weight * min_distances ** 2 + remaining_dists_squared, nearest_entries

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

# Given two values of theta and a given group, transform theta1 (via the group
# action) such that the euclidean interpolation from theta0 to theta1 is the
# geodesic interpolation on SO(2) / G.
def WrapTheta2d(G, theta0, theta1):
    theta_step = 2 * np.pi / G.order()
    theta_thresh = theta_step / 2
    while theta1 - theta0 > theta_thresh:
        theta1 -= theta_step
    while theta1 - theta0 < -theta_thresh:
        theta1 += theta_step
    return theta1

# Given a list of thetas theta0s and a 2d matrix of corresponding nearest entries
# nearest_thetas, return nearest_thetas wrapped appropriately.
def WrapTheta2dVectorized(G, theta0s, nearest_thetas):
    theta_step = 2 * np.pi / G.order()
    theta_thresh = theta_step / 2

    # Compute the difference
    diff = nearest_thetas - theta0s[:, np.newaxis]  # Shape: (len(q1s), len(nearest_thetas))

    # Wrap the differences to the range [-theta_thresh, theta_thresh]
    wrapped_diff = (diff + theta_thresh) % theta_step - theta_thresh

    # Update nearest_thetas
    nearest_thetas = theta0s[:, np.newaxis] + wrapped_diff

    return nearest_thetas

def UnwrapToContinuousPath2d(G, path, symmetry_idx):
    if len(path) == 0:
        return []

    new_path = [path[0][0].copy(), path[0][1].copy()]
    for start, end in path[1:]:
        dtheta_old = end[symmetry_idx] - start[symmetry_idx]
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

        new_path.append(end.copy())
        new_path[-1][symmetry_idx] = theta_next
        new_path[-1][symmetry_idx] = WrapTheta2d(G, new_path[-2][symmetry_idx], new_path[-1][symmetry_idx])

        # while new_path[-1][symmetry_idx] >= new_path[-2][symmetry_idx] + (np.pi / G.order()):
        #     new_path[-1][symmetry_idx] -= 2 * np.pi / G.order()
        # while new_path[-1][symmetry_idx] <= new_path[-2][symmetry_idx] - (np.pi / G.order()):
        #     new_path[-1][symmetry_idx] += 2 * np.pi / G.order()

        dtheta_new = new_path[-1][symmetry_idx] - new_path[-2][symmetry_idx]
        if np.abs(dtheta_new - dtheta_old) > 1e-5:
            print(dtheta_old, dtheta_new)
            import pdb
            pdb.set_trace()

    return new_path

if __name__ == "__main__":
    G = src.symmetry.CyclicGroupSO2(3)
    D = SO2DistanceSq(G, 3, 2)
    q1 = np.array([0, 0, 0])
    q2 = np.array([0, 0, 2 * np.pi / 3])
    # print(D(q1, q2))
    # print(D.pairwise([q1, q2]))
    q3 = np.array([1, 0, 2 * np.pi / 3])
    print()
    print(D.pairwise([q1, q2, q3]))

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

    S = SO2SampleUniform(G, 3, 2, [-1, -1, 0], [1, 1, 0])
    print()
    print(S(1))
    print(S(5))