import sys
sys.path.append("..")

import time
import numpy as np

import src.planners.imacs as imacs
import src.planners.rrt as rrt
import src.planners.shortcut as shortcut
import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory
)

options = rrt.RRTOptions(max_vertices=2e4, max_iters=1e4, step_size=1.0, goal_sample_frequency=0.01)
shortcut_options = shortcut.ShortcutOptions(max_iters=1e3)

meshcat = StartMeshcat()

# n_sides = 8
n_sides = 2

use_birrt = True
n_copies = 2
# q0 = np.array([0.01, 0.01, 0,
#                19.9, 0.01, 0,
#                19.9, 19.9, 0,
#                0.01, 19.9, 0])[:3*n_copies]
# q1 = np.array([19.9, 0.01, np.pi,
#                19.9, 19.9, np.pi,
#                0.01, 19.9, np.pi,
#                0.01, 0.01, np.pi])[:3*n_copies]

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(n_sides, limits, 50, 0.75, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params, n_copies=n_copies)

limits_lower = [limits[0][0], limits[1][0], 0] * n_copies
limits_upper = [limits[0][1], limits[1][1], 0] * n_copies

cspace_dim = 3 * n_copies
symmetry_indices = symmetry_indices = list(range(2, 3*n_copies, 3))

G = symmetry.CyclicGroupSO2(n_sides)
Sampler = imacs.SO2SampleUniform(G, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric = imacs.SO2DistanceSqMultiple(G, cspace_dim, symmetry_indices)
Interpolator = imacs.SO2InterpolateMultiple(G, cspace_dim, symmetry_indices)
if use_birrt:
    planner = rrt.BiRRT(Sampler, Metric, Interpolator, CollisionChecker, options)
else:
    planner = rrt.RRT(Sampler, Metric, Interpolator, CollisionChecker, options)
shortcutter = shortcut.Shortcut(Metric, Interpolator, CollisionChecker, shortcut_options)

qs = []
np.random.seed(0)
while len(qs) < 2:
    q = Sampler(1).flatten()
    print(len(qs))
    if len(qs) < 1:
        q[0:3] = [ 2.51170621,  4.14485756, -2.81821468]
    if CollisionChecker.CheckConfigCollisionFree(q):
        qs.append(q)
q0, q1 = qs
print(q0, q1)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.plant().SetPositions(plant_context, q0)
diagram.ForcedPublish(diagram_context)

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

np.random.seed(0)
path = planner.plan(q0, q1, verbose=True)
path = shortcutter.shortcut(path, verbose=True)
path = imacs.UnwrapToContinuousPath2dMultiple(G, path, symmetry_indices)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

one_object_metric = imacs.SO2DistanceSq(G, 3, 2)
print("SE(2)/G path length:", Metric.path_length(path))
t_scaling = 2/4
times = [t_scaling * np.sqrt(np.max([one_object_metric(path[i-1][j-2:j+1], path[i][j-2:j+1])[0] for j in symmetry_indices])) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj1 = CompositeTrajectory.AlignAndConcatenate(segments)

traj_dt = traj1.end_time() - traj1.start_time()
dt = 1/60
for t in np.linspace(traj1.start_time(), traj1.end_time(), int(traj_dt / dt)):
    diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Now compare to the plan without symmetries

G = symmetry.CyclicGroupSO2(1)
Sampler = imacs.SO2SampleUniform(G, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric = imacs.SO2DistanceSqMultiple(G, cspace_dim, symmetry_indices)
Interpolator = imacs.SO2InterpolateMultiple(G, cspace_dim, symmetry_indices)
if use_birrt:
    planner = rrt.BiRRT(Sampler, Metric, Interpolator, CollisionChecker, options)
else:
    planner = rrt.RRT(Sampler, Metric, Interpolator, CollisionChecker, options)
shortcutter = shortcut.Shortcut(Metric, Interpolator, CollisionChecker, shortcut_options)

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

np.random.seed(0)
path = planner.plan(q0, q1, verbose=True)
path = shortcutter.shortcut(path, verbose=True)
path = imacs.UnwrapToContinuousPath2dMultiple(G, path, symmetry_indices)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

one_object_metric = imacs.SO2DistanceSq(G, 3, 2)
print("Baseline path length:", Metric.path_length(path))
times = [t_scaling * np.sqrt(np.max([one_object_metric(path[i-1][j-2:j+1], path[i][j-2:j+1])[0] for j in symmetry_indices])) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj2 = CompositeTrajectory.AlignAndConcatenate(segments)

traj_dt = traj1.end_time() - traj1.start_time()
dt = 1/60
for t in np.linspace(traj2.start_time(), traj2.end_time(), int(traj_dt / dt)):
    diagram.plant().SetPositions(plant_context, traj2.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Alternate visualizing each one

while True:
    for traj, name in zip([traj1, traj2], ["symmetry aware", "symmetry unaware"]):
        time.sleep(3)
        diagram.plant().SetPositions(plant_context, traj.value(traj.start_time()).flatten())
        diagram.ForcedPublish(diagram_context)
        time.sleep(3)

        traj_dt = traj.end_time() - traj.start_time()
        dt = 1/60
        print(name)
        for t in np.linspace(traj.start_time(), traj.end_time(), int(traj_dt / dt)):
            diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(diagram_context)
            time.sleep(dt)