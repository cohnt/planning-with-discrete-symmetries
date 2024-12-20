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

options = rrt.RRTOptions(max_vertices=1e3, max_iters=1e4, goal_sample_frequency=0.05)
shortcut_options = shortcut.ShortcutOptions(max_iters=1e2)

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(3, limits, 200, 1.25, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

G = symmetry.CyclicGroupSO2(3)
Sampler = imacs.SO2SampleUniform(G, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric = imacs.SO2DistanceSq(G, 3, 2)
Interpolator = imacs.SO2Interpolate(G, 3, 2)
planner = rrt.BiRRT(Sampler, Metric, Interpolator, CollisionChecker, options)
shortcutter = shortcut.Shortcut(Metric, Interpolator, CollisionChecker, shortcut_options)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

# Visualize a random plan
q0 = np.array([0.01, 0.01, 0])
q1 = np.array([19.9, 19.9, np.pi])

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

np.random.seed(0)
path = planner.plan(q0, q1, verbose=True)
path = shortcutter.shortcut(path, verbose=True)
path = imacs.UnwrapToContinuousPath2d(G, path, 2)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

print("SE(2)/G path length:", Metric.path_length(path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj1 = CompositeTrajectory.AlignAndConcatenate(segments)

dt = traj1.end_time() - traj1.start_time()
dt /= 400
for t in np.linspace(traj1.start_time(), traj1.end_time(), 400):
    diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Now compare to the plan without symmetries

G = symmetry.CyclicGroupSO2(1)
Sampler = imacs.SO2SampleUniform(G, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric = imacs.SO2DistanceSq(G, 3, 2)
Interpolator = imacs.SO2Interpolate(G, 3, 2)
planner = rrt.BiRRT(Sampler, Metric, Interpolator, CollisionChecker, options)
shortcutter = shortcut.Shortcut(Metric, Interpolator, CollisionChecker, shortcut_options)

q0 = np.array([0.01, 0.01, 0])
q1 = np.array([19.9, 19.9, np.pi])

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

np.random.seed(0)
path = planner.plan(q0, q1, verbose=True)
path = shortcutter.shortcut(path, verbose=True)
path = imacs.UnwrapToContinuousPath2d(G, path, 2)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

print("SE(2)/G path length:", Metric.path_length(path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj2 = CompositeTrajectory.AlignAndConcatenate(segments)

dt = traj1.end_time() - traj1.start_time()
dt /= 400
for t in np.linspace(traj2.start_time(), traj2.end_time(), 400):
    diagram.plant().SetPositions(plant_context, traj2.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Alternate visualizing each one

while True:
    for traj in [traj1, traj2]:
        time.sleep(3)
        dt = traj.end_time() - traj.start_time()
        dt /= 400
        for t in np.linspace(traj.start_time(), traj.end_time(), 400):
            diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(diagram_context)
            time.sleep(dt)