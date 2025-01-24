import sys
sys.path.append("..")

import time
import numpy as np

import src.planners.imacs as imacs
import src.planners.rrt as rrt
import src.planners.shortcut as shortcut
import src.symmetry as symmetry
import src.visualization as visualization
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory,
    Rgba
)

options = rrt.RRTOptions(max_vertices=1e3, max_iters=1e4, goal_sample_frequency=0.05, stop_at_goal=False)
shortcut_options = shortcut.ShortcutOptions(max_iters=1e2)
random_seed = 0

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(3, limits, 200, 1.25, random_seed)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)

G = symmetry.CyclicGroupSO2(3)
Sampler = imacs.SO2SampleUniform(G, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric = imacs.SO2DistanceSq(G, 3, 2)
Interpolator = imacs.SO2Interpolate(G, 3, 2)
planner = rrt.RRT(Sampler, Metric, Interpolator, CollisionChecker, options)
shortcutter = shortcut.Shortcut(Metric, Interpolator, CollisionChecker, shortcut_options)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

# Visualize a random plan
q0 = np.array([0.01, 0.01, 0])
q1 = np.array([19.9, 19.9, np.pi])

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

np.random.seed(random_seed)
path = planner.plan(q0, q1, verbose=True)
# path = shortcutter.shortcut(path, verbose=True)
path = imacs.UnwrapToContinuousPath2d(G, path, 2)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

print("SE(2)/G path length:", Metric.path_length(path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj1 = CompositeTrajectory.AlignAndConcatenate(segments)

# dt = traj1.end_time() - traj1.start_time()
# dt /= 400
# for t in np.linspace(traj1.start_time(), traj1.end_time(), 400):
#     diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
#     diagram.ForcedPublish(diagram_context)
#     time.sleep(dt)

# # Now compare to the plan without symmetries

# G2 = symmetry.CyclicGroupSO2(1)
# Sampler2 = imacs.SO2SampleUniform(G2, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
# Metric2 = imacs.SO2DistanceSq(G2, 3, 2)
# Interpolator2 = imacs.SO2Interpolate(G2, 3, 2)
# planner2 = rrt.RRT(Sampler2, Metric2, Interpolator2, CollisionChecker, options)
# shortcutter2 = shortcut.Shortcut(Metric2, Interpolator2, CollisionChecker, shortcut_options)

# q0 = np.array([0.01, 0.01, 0])
# q1 = np.array([19.9, 19.9, np.pi])

# assert CollisionChecker.CheckConfigCollisionFree(q0)
# assert CollisionChecker.CheckConfigCollisionFree(q1)

# np.random.seed(random_seed)
# path2 = planner2.plan(q0, q1, verbose=True)
# # path2 = shortcutter2.shortcut(path2, verbose=True)
# path2 = imacs.UnwrapToContinuousPath2d(G2, path2, 2)

# for i in range(1, len(path2)):
#     # print(path2[i-1][2], path2[i][2])
#     assert np.abs(path2[i-1][2] - path2[i][2]) <= np.pi

# print("Baseline path length:", Metric2.path_length(path2))
# t_scaling = 1/4
# times = [t_scaling * np.sqrt(Metric2(path2[i-1], path2[i])[0]) for i in range(1, len(path2))]
# segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path2[i-1], path2[i]]).T) for i in range(1, len(path2))]
# traj2 = CompositeTrajectory.AlignAndConcatenate(segments)

# dt = traj1.end_time() - traj1.start_time()
# dt /= 400
# for t in np.linspace(traj2.start_time(), traj2.end_time(), 400):
#     diagram.plant().SetPositions(plant_context, traj2.value(t).flatten())
#     diagram.ForcedPublish(diagram_context)
#     time.sleep(dt)

# Alternate visualizing each one

# while True:
#     for traj in [traj1, traj2]:
#         time.sleep(3)
#         dt = traj.end_time() - traj.start_time()
#         dt /= 400
#         for t in np.linspace(traj.start_time(), traj.end_time(), 400):
#             diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
#             diagram.ForcedPublish(diagram_context)
#             time.sleep(dt)

visualization.draw_graph(meshcat, planner.tree, [0,1], path="rrt", color=Rgba(0, 0, 0, 1), linewidth=2.0)

# Check RRT*
import src.planners.star as star
rrt_star_options = star.RRTStarOptions(connection_radius=20.0, connection_k=12, mode="radius")
rrt_star = star.RRTStar(planner, rrt_star_options)

new_path = rrt_star.return_plan()
new_path = imacs.UnwrapToContinuousPath2d(G, new_path, 2)

for i in range(1, len(new_path)):
    # print(new_path[i-1][2], new_path[i][2])
    assert np.abs(new_path[i-1][2] - new_path[i][2]) <= np.pi

print("SE(2)/G RRT* path length:", Metric.path_length(new_path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(new_path[i-1], new_path[i])[0]) for i in range(1, len(new_path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([new_path[i-1], new_path[i]]).T) for i in range(1, len(new_path))]
new_traj = CompositeTrajectory.AlignAndConcatenate(segments)

visualization.draw_graph(meshcat, planner.tree, [0,1], path="rrt_star", color=Rgba(0, 1, 0, 1))

# Alternate visualizing RRT and RRT*

while True:
    for traj, name in zip([traj1, new_traj], ["RRT", "RRT*"]):
        time.sleep(3)
        print("%s path" % name)
        dt = traj.end_time() - traj.start_time()
        dt /= 400
        for t in np.linspace(traj.start_time(), traj.end_time(), 400):
            diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(diagram_context)
            time.sleep(dt)