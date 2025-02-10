import sys
sys.path.append("..")

import time
import numpy as np

import src.asymptotic_optimality_parameters as asymptotic
import src.planners.imacs as imacs
import src.planners.prm as prm
import src.symmetry as symmetry
import src.worlds.path_planning_3d as path_planning_3d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory
)

options = prm.PRMOptions(max_vertices=5e2, neighbor_k=12, neighbor_radius=5e0, neighbor_mode="k")
random_seed = 0

meshcat = StartMeshcat()

limits = [[0, 10], [0, 10], [0, 10]]
G = symmetry.TetrahedralGroup()
params = path_planning_3d.SetupParams(G, True, limits, 150, 0.7, 0)
diagram, CollisionChecker = path_planning_3d.build_env(meshcat, params)

c_free_volume = 1
c_free_volume *= limits[0][1] - limits[0][0]
c_free_volume *= limits[1][1] - limits[1][0]
c_free_volume *= asymptotic.s1_volume()
print("Symmetry-Aware PRM* Minimum Radius:", asymptotic.radius_prm(3, c_free_volume / 3))
print("Symmetry-Unaware PRM* Minimum Radius:", asymptotic.radius_prm(3, c_free_volume))
print("KNN-PRM* Minimum k:", asymptotic.knn_prm(3))

options.neighbor_radius = asymptotic.radius_prm(3, c_free_volume)
options.neighbor_k = asymptotic.knn_prm(3)

sampler_limits_lower = np.zeros(12)
sampler_limits_upper = np.zeros(12)
sampler_limits_lower[-3:] = [limits[0][0], limits[1][0], limits[2][0]]
sampler_limits_upper[-3:] = [limits[0][1], limits[1][1], limits[2][1]]
Sampler = imacs.SO3SampleUniform(G, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
Metric = imacs.SO3DistanceSq(G, 12, 0)
Interpolator = imacs.SO3Interpolate(G, 12, 0)
CollisionCheckerWrapper = imacs.SO3CollisionCheckerWrapper(CollisionChecker, 12, 0)
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionCheckerWrapper, options)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)

np.random.seed(random_seed)
roadmap.build(verbose=True)

fname = "check_prm_3d_symmetric.pkl"
roadmap.save(fname)
roadmap = prm.PRM.load(fname, CollisionCheckerWrapper)

# Visualize a random plan
q0 = np.append(np.eye(3).flatten(), [0.01, 0.01, 0.01])
q1 = np.append(np.eye(3).flatten(), [9.99, 9.99, 9.99])

assert CollisionCheckerWrapper.CheckConfigCollisionFree(q0)
assert CollisionCheckerWrapper.CheckConfigCollisionFree(q1)

path = roadmap.plan(q0, q1)
path = imacs.UnwrapToContinuousPathSO3(G, path, 0)
traj1 = imacs.SO3PathToDrakeSlerpTraj(Metric, path, 0)

print("SE(3)/G path length:", Metric.path_length(path))

# Baseline

G2 = symmetry.CyclicGroupSO3(1)
Sampler = imacs.SO3SampleUniform(G2, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
Metric = imacs.SO3DistanceSq(G2, 12, 0)
Interpolator = imacs.SO3Interpolate(G2, 12, 0)
CollisionCheckerWrapper = imacs.SO3CollisionCheckerWrapper(CollisionChecker, 12, 0)
roadmap2 = prm.PRM(Sampler, Metric, Interpolator, CollisionCheckerWrapper, options)

np.random.seed(random_seed)
roadmap2.build(verbose=True)

fname = "check_prm_3d_baseline.pkl"
roadmap2.save(fname)
roadmap2 = prm.PRM.load(fname, CollisionCheckerWrapper)

path = roadmap2.plan(q0, q1)
path = imacs.UnwrapToContinuousPathSO3(G2, path, 0)
traj2 = imacs.SO3PathToDrakeSlerpTraj(Metric, path, 0)

print("Baseline path length:", Metric.path_length(path))

# Compare

n_steps = 400
extra_time_scaling = 0.25

dt = traj1.end_time() - traj1.start_time()
dt /= n_steps
dt *= extra_time_scaling
while True:
    for t in np.linspace(traj1.start_time(), traj1.end_time(), n_steps):
        diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
        diagram.ForcedPublish(diagram_context)
        time.sleep(dt)

    for t in np.linspace(traj2.start_time(), traj2.end_time(), n_steps):
        diagram.plant().SetPositions(plant_context, traj2.value(t).flatten())
        diagram.ForcedPublish(diagram_context)
        time.sleep(dt)