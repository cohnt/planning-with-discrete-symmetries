import sys
sys.path.append("..")

import time
import numpy as np

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

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20], [0, 20]]
G = symmetry.TetrahedralGroup()
# params = path_planning_3d.SetupParams(G, True, limits, 300, 0.45, 0)
params = path_planning_3d.SetupParams(G, True, limits, 100, 0.45, 0)
diagram, CollisionChecker = path_planning_3d.build_env(meshcat, params)

sampler_limits_lower = np.zeros(12)
sampler_limits_upper = np.zeros(12)
sampler_limits_lower[-3:] = [limits[0][0], limits[1][0], limits[2][0]]
sampler_limits_upper[-3:] = [limits[0][1], limits[1][1], limits[2][1]]
Sampler = imacs.SO3SampleUniform(G, 12, 0, sampler_limits_lower, sampler_limits_upper)
Metric = imacs.SO3DistanceSq(G, 12, 0)
Interpolator = imacs.SO3Interpolate(G, 12, 0)
CollisionCheckerWrapper = imacs.SO3CollisionCheckerWrapper(CollisionChecker, 12, 0)
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionCheckerWrapper, options)

np.random.seed(0)
roadmap.build()

fname = "check_prm_3d_symmetric.pkl"
roadmap.save(fname)
roadmap = prm.PRM.load(fname, CollisionCheckerWrapper)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

# Visualize a random plan
q0 = np.append(np.eye(3).flatten(), [0.01, 0.01, 0])
q1 = np.append(np.eye(3).flatten(), [19.9, 19.9, 19.9])

assert CollisionCheckerWrapper.CheckConfigCollisionFree(q0)
assert CollisionCheckerWrapper.CheckConfigCollisionFree(q1)

path = roadmap.plan(q0, q1)

path = imacs.UnwrapToContinuousPathSO3(G, path, 0)

# raw_path = [path[0][0]]
# for foo, bar in path:
#     raw_path.append(bar)
# path = raw_path

traj1 = imacs.SO3PathToDrakeSlerpTraj(Metric, path, 0)

n_steps = 400
extra_time_scaling = 0.25

dt = traj1.end_time() - traj1.start_time()
dt /= n_steps
dt *= extra_time_scaling
while True:
    for t in np.linspace(traj1.start_time(), traj1.end_time(), n_steps):
        # print(t, traj1.value(t))
        diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
        diagram.ForcedPublish(diagram_context)
        time.sleep(dt)