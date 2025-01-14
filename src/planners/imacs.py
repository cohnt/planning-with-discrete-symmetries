import numpy as np
import src.symmetry

from pydrake.all import (
    RotationMatrix,
    Quaternion,
    PiecewiseQuaternionSlerp,
    StackedTrajectory,
    PiecewisePolynomial,
    CompositeTrajectory
)
from scipy.stats import special_ortho_group

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

def quaternion_to_so3(qs_in):
    qs = qs_in.copy()
    # Stored as [w, x, y, z]. Must be a numpy array
    single = False
    if len(qs.shape) == 1:
        single = True
        qs = qs.reshape(1, 4)
    mats = []
    for q in qs:
        mats.append(RotationMatrix(Quaternion(*q)).matrix())
    if single:
        return mats[0]
    else:
        return np.array(mats)

def so3_to_quaternion(ms_in):
    ms = ms_in.copy()
    single = False
    if len(ms.shape) == 2:
        single = True
        ms = ms.reshape(1, 3, 3)
    qs = []
    for m in ms:
        qs.append(RotationMatrix(m).ToQuaternion().wxyz())
    if single:
        return qs[0]
    else:
        return np.array(qs)


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
        nearest_thetas = WrapTheta2dVectorized(self.G, np_q1s[:,self.symmetry_dof_start], nearest_thetas_old)

        # Assign theta values correctly
        nearest_entries = np.tile(np_q2s, (len(q1s), 1, 1))
        nearest_entries[:, :, self.symmetry_dof_start] = nearest_thetas

        return self.symmetry_weight * min_distances ** 2 + remaining_dists_squared, nearest_entries

class SO3DistanceSq(DistanceSq):
    def __call__(self, q1, q2):
        assert len(q1) == self.ambient_dim
        assert len(q2) == self.ambient_dim

        symmetry_dof_end = self.symmetry_dof_start + 9

        m1 = q1[self.symmetry_dof_start:symmetry_dof_end].reshape(3,3)
        m2 = q2[self.symmetry_dof_start:symmetry_dof_end].reshape(3,3)
        m2_prime = self.G.orbit(m2)
        distances = rotation_distance_so3(m1, m2_prime)
        idx = np.argmin(distances)
        symmetry_distance = distances[idx]
        remaining_distance_sq = np.sum((np.delete(q1, slice(self.symmetry_dof_start, symmetry_dof_end)) - np.delete(q2, slice(self.symmetry_dof_start, symmetry_dof_end))) ** 2)
        total_distance_sq = self.symmetry_weight * symmetry_distance**2 + remaining_distance_sq
        q_out = q2.copy()
        q_out[self.symmetry_dof_start:symmetry_dof_end] = m2_prime[idx].flatten()
        return total_distance_sq, q_out

    def pairwise(self, q1s, q2s=None):
        if q2s is None:
            q2s = q1s

        np_q1s = np.asarray(q1s)
        np_q2s = np.asarray(q2s)

        symmetry_dof_end = self.symmetry_dof_start + 9

        # Euclidean components
        remaining_q1s = np.delete(np_q1s, slice(self.symmetry_dof_start, symmetry_dof_end), axis=1)
        remaining_q2s = np.delete(np_q2s, slice(self.symmetry_dof_start, symmetry_dof_end), axis=1)
        remaining_dists_squared = (
            np.sum(remaining_q1s**2, axis=1, keepdims=True)  # Shape (n, 1)
            + np.sum(remaining_q2s**2, axis=1)              # Shape (m,)
            - 2 * np.dot(remaining_q1s, remaining_q2s.T)    # Shape (n, m)
        )

        R1s = np_q1s[:,self.symmetry_dof_start:symmetry_dof_end].reshape(-1, 3, 3)
        R2s = np_q2s[:,self.symmetry_dof_start:symmetry_dof_end].reshape(-1, 3, 3)
        orbits = np.array([self.G.orbit(R2) for R2 in R2s])  # Shape (m, orbit_len, 3, 3)

        # Compute pairwise R = m1 @ m2^T for all combinations of matrices1 and orbits
        R = R1s[:, None, None] @ np.transpose(orbits, (0, 1, 3, 2))  # Shape: (N, M, orbit_size, 3, 3)

        # Compute the cosine of the angle for each pair
        cos_theta = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2  # Shape: (N, M, orbit_size)

        # Compute the angles (distances)
        theta = np.arccos(np.clip(cos_theta, -1, 1))  # Shape: (N, M, orbit_size)

        # Find the minimum distance and corresponding matrix for each (i, j) pair
        min_indices = np.argmin(theta, axis=2)  # Shape: (N, M)
        distances = np.min(theta, axis=2)  # Shape: (N, M)

        # Extract the closest matrices from the orbit
        N, M = min_indices.shape
        orbit_size = orbits.shape[1]

        # Prepare indices for advanced indexing
        orbit_indices = np.arange(orbit_size)
        min_indices_expanded = min_indices[:, :, None, None, None]  # Shape: (N, M, 1, 1)

        orbits_repeated = np.repeat(orbits[None, :, :, :, :], N, axis=0)  # Shape: (N, M, orbit_size, 3, 3)
        orbits_repeated = np.repeat(orbits[None, :, :, :, :], R1s.shape[0], axis=0)  # Shape: (N, M, orbit_size, 3, 3)

        closest_orbits = np.take_along_axis(orbits_repeated, min_indices_expanded, axis=2).squeeze(2)  # Shape: (N, M, 3, 3)

        # Combine with the Euclidean distance components
        distances = self.symmetry_weight * distances ** 2 + remaining_dists_squared

        # Fix how nearest entries are listed
        nearest_entries = np.tile(np_q2s, (len(q1s), 1, 1))

        nearest_entries[:, :, self.symmetry_dof_start:symmetry_dof_end] = closest_orbits.reshape(len(q1s), len(q2s), 9)

        return distances, nearest_entries

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

class SO3Interpolate(Interpolate):
    def __call__(self, q1, q2, t):
        assert 0 <= t <= 1

        # Special logic needed for numerical stability.
        if t < 1e-10:
            return q1
        elif 1 - t < 1e-10:
            return q2

        symmetry_dof_end = self.symmetry_dof_start + 9
        mat1 = q1[self.symmetry_dof_start:symmetry_dof_end].reshape(3,3)
        mat2 = q2[self.symmetry_dof_start:symmetry_dof_end].reshape(3,3)

        quat1 = RotationMatrix(mat1).ToQuaternion().wxyz()
        quat2 = RotationMatrix(mat2).ToQuaternion().wxyz()
        
        cos_theta = np.dot(quat1, quat2)
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        t1 = np.sin((1 - t) * theta) / np.sin(theta)
        t2 = np.sin(t * theta) / np.sin(theta)
        quat_out = t1 * quat1 + t2 * quat2
        if cos_theta < 0:
            quat_out *= -1

        new_mat = RotationMatrix(Quaternion(*quat_out)).matrix()
        q_out = t * q2 + (1 - t) * q1
        q_out[self.symmetry_dof_start:symmetry_dof_end] = new_mat.flatten()
        return q_out

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

class SO3SampleUniform(SampleUniform):
    def __init__(self, G, ambient_dim, symmetry_dof_start, limits_lower, limits_upper, random_seed=0):
        super().__init__(G, ambient_dim, symmetry_dof_start, limits_lower, limits_upper)
        self.limits_lower[self.symmetry_dof_start] = -1
        self.limits_upper[self.symmetry_dof_start] = 1
        # These will be ignored, since we sample from SO(3) in a special way. But we need them to be finite,
        # so that we can call np.random.uniform, before substituting the new values in.

        # Keep consistency between the random seed we use in numpy and scipy.
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, n):
        qs = np.random.uniform(low=self.limits_lower, high=self.limits_upper, size=(n, self.ambient_dim))
        symmetry_dof_end = self.symmetry_dof_start + 9
        for i in range(len(qs)):
            qs[i,self.symmetry_dof_start:symmetry_dof_end] = special_ortho_group.rvs(3, random_state=self.rng).flatten()
        return qs

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

def UnwrapToContinuousPathSO3(G, path, symmetry_dof_start):
    if len(path) == 0:
        return []

    symmetry_dof_end = symmetry_dof_start + 9

    new_path = [path[0][0].copy(), path[0][1].copy()]
    for start, end in path[1:]:
        m1 = new_path[-1][symmetry_dof_start:symmetry_dof_end].reshape(3,3)
        m2 = start[symmetry_dof_start:symmetry_dof_end].reshape(3,3)
        
        if not G.equivalent(m1, m2):
            import pdb
            pdb.set_trace()

        assert G.equivalent(m1, m2)

        # We will right-multiply the start and end matrices by this matrix
        dmat = m2.T @ m1
        new_end = end.copy()
        new_mat = end[symmetry_dof_start:symmetry_dof_end].reshape(3,3) @ dmat
        new_end[symmetry_dof_start:symmetry_dof_end] = new_mat.flatten()
        new_path.append(new_end)

    return new_path

def SO3PathToDrakeSlerpTraj(Metric, path, symmetry_dof_start):
    assert len(path) > 1
    times = [np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]

    symmetry_dof_end = symmetry_dof_start + 9
    mats = np.asarray(path)[:,symmetry_dof_start:symmetry_dof_end]
    mats = mats.reshape(len(path), 3, 3)
    slerp_breaks = np.append(0, np.cumsum(times))
    slerp_traj = PiecewiseQuaternionSlerp(slerp_breaks, mats)

    full_traj = StackedTrajectory(rowwise=True)
    if symmetry_dof_start > 0:
        segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1][:symmetry_dof_start], path[i][:symmetry_dof_start]]).T) for i in range(1, len(path))]
        top_traj = CompositeTrajectory.AlignAndConcatenate(segments)
        full_traj.Append(top_traj)

    full_traj.Append(slerp_traj)

    if symmetry_dof_end < len(path[0]):
        segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1][symmetry_dof_end:], path[i][symmetry_dof_end:]]).T) for i in range(1, len(path))]
        top_traj = CompositeTrajectory.AlignAndConcatenate(segments)
        full_traj.Append(top_traj)

    return full_traj

class SO3CollisionCheckerWrapper():
    def __init__(self, CollisionChecker, ambient_dim, symmetry_dof_start):
        self.CollisionChecker = CollisionChecker
        self.ambient_dim = ambient_dim
        self.symmetry_dof_start = symmetry_dof_start

    def _remap_qs(self, qs):
        new_qs = np.zeros((qs.shape[0], qs.shape[1] - 5))
        new_qs[:,:self.symmetry_dof_start] = qs[:,:self.symmetry_dof_start]
        new_qs[:,self.symmetry_dof_start:self.symmetry_dof_start+4] = so3_to_quaternion(qs[:,self.symmetry_dof_start:self.symmetry_dof_start+9].reshape(-1,3,3))
        new_qs[:,self.symmetry_dof_start+4:] = qs[:,self.symmetry_dof_start+9:]
        return new_qs

    def CheckConfigCollisionFree(self, q):
        return self.CheckConfigsCollisionFree(np.array([q]))

    def CheckConfigsCollisionFree(self, qs):
        # 9 values needed for SO(3) matrix, but only 4 for the quaternion
        return self.CollisionChecker.CheckConfigsCollisionFree(self._remap_qs(qs))

    def CheckEdgeCollisionFreeParallel(self, q1, q2):
        new_q1, new_q2 = self._remap_qs(np.array([q1, q2]))
        return self.CollisionChecker.CheckEdgeCollisionFreeParallel(new_q1, new_q2)

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

    print()

    G = src.symmetry.CyclicGroupSO3(4)
    D = SO3DistanceSq(G, 12, 3)
    q1 = np.append(np.zeros(3), np.eye(3).flatten())
    q2 = np.append(np.zeros(3), [
        1, 0, 0,
        0, 0, 1,
        0, -1, 0
    ]) # Rotation about x axis by 90 degrees
    print(D(q1, q2)) # Should be zero, since it's a square pyramid
    q3 = np.append(np.array([0, 1, 0]), [
        -1, 0, 0,
        0, -1, 0,
        0, 0, 1
    ]) # Rotation about z axis by 180 degrees, and slight translation
    print(1**2 + np.pi**2, D(q1, q3)) # Should be 1^2 + pi^2 (about 10.8)

    I = SO3Interpolate(G, 12, 3)
    print()
    print(I(q1, q3, 0))
    print(I(q1, q3, 0.5))
    print(I(q1, q3, 1))
    print(I(q1, q2, 0)[3:].reshape(3,3))
    print(I(q1, q2, 0.5)[3:].reshape(3,3))
    print(I(q1, q2, 1)[3:].reshape(3,3))

    S = SO3SampleUniform(G, 12, 3, -np.ones(12), np.ones(12))
    print()
    q = S(1)
    print(q)
    print(q[0, 3:].reshape(3,3) @ q[0, 3:].reshape(3,3).T)
    print(S(5))