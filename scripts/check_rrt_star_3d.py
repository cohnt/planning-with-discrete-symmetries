import sys
sys.path.append("..")

import time
import numpy as np

import src.asymptotic_optimality_parameters as aop
import src.planners.imacs as imacs
import src.planners.rrt as rrt
import src.symmetry as symmetry
import src.visualization as visualization
import src.worlds.path_planning_3d as path_planning_3d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory,
    Rgba
)

options = rrt.RRTOptions(max_vertices=250, max_iters=1e4, goal_sample_frequency=0.05, stop_at_goal=False, step_size=5.0)
random_seed = 0

meshcat = StartMeshcat()

G = symmetry.OctahedralGroup()
limits = [[0, 10], [0, 10], [0, 10]]
params = path_planning_3d.SetupParams(G, False, limits, 150, 0.7, 0)
diagram, CollisionChecker = path_planning_3d.build_env(meshcat, params)
CollisionCheckerWrapper = imacs.SO3CollisionCheckerWrapper(CollisionChecker, 12, 0)

sampler_limits_lower = np.zeros(12)
sampler_limits_upper = np.zeros(12)
sampler_limits_lower[-3:] = [limits[0][0], limits[1][0], limits[2][0]]
sampler_limits_upper[-3:] = [limits[0][1], limits[1][1], limits[2][1]]

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)

Sampler = imacs.SO3SampleUniform(G, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
Metric = imacs.SO3DistanceSq(G, 12, 0)
Interpolator = imacs.SO3Interpolate(G, 12, 0)
planner = rrt.RRT(Sampler, Metric, Interpolator, CollisionCheckerWrapper, options)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

# Visualize a random plan
M = np.eye(3)
M[0,0] = M[1,1] = -1
q0 = np.append(np.eye(3).flatten(), [9.99, 0.01, 0.01])
q1 = np.append(M.flatten(), [0.01, 9.99, 9.99])

assert CollisionCheckerWrapper.CheckConfigCollisionFree(q0)
assert CollisionCheckerWrapper.CheckConfigCollisionFree(q1)

np.random.seed(random_seed)
path = planner.plan(q0, q1, verbose=True)
path = imacs.UnwrapToContinuousPathSO3(G, path, 0)
traj1 = imacs.SO3PathToDrakeSlerpTraj(Metric, path, 0)

print("SE(3)/G path length:", Metric.path_length(path))

# Now compare to the plan without symmetries

G2 = symmetry.CyclicGroupSO3(1)
Sampler2 = imacs.SO3SampleUniform(G2, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
Metric2 = imacs.SO3DistanceSq(G2, 12, 0)
Interpolator2 = imacs.SO3Interpolate(G2, 12, 0)
planner2 = rrt.RRT(Sampler2, Metric2, Interpolator2, CollisionCheckerWrapper, options)

assert CollisionCheckerWrapper.CheckConfigCollisionFree(q0)
assert CollisionCheckerWrapper.CheckConfigCollisionFree(q1)

np.random.seed(random_seed)
path2 = planner2.plan(q0, q1, verbose=True)
path2 = imacs.UnwrapToContinuousPathSO3(G2, path2, 0)
traj2 = imacs.SO3PathToDrakeSlerpTraj(Metric2, path2, 0)

print("Baseline path length:", Metric2.path_length(path2))

# visualization.draw_graph(meshcat, planner2.tree, [0,1], path="baseline-rrt", color=Rgba(0.5, 0, 0, 1), linewidth=4.0)

# Check RRT*
import src.planners.star as star

c_space_dimension = 6
c_space_volume = np.prod(np.array(limits)[:,1] - np.array(limits)[:,0]) * aop.so3_volume()
rrt_star_radius_original = aop.radius_rrt(c_space_dimension, c_space_volume)
rrt_star_radius_quotient = aop.radius_rrt(c_space_dimension, c_space_volume / G.order())

rrt_star_options = star.RRTStarOptions(connection_radius=rrt_star_radius_original, connection_k=12, mode="radius")

rrt_star = star.RRTStar(planner, rrt_star_options)
new_path = rrt_star.return_plan()
new_path = imacs.UnwrapToContinuousPathSO3(G, new_path, 0)
new_traj = imacs.SO3PathToDrakeSlerpTraj(Metric, new_path, 0)

print("SE(2)/G RRT* path length:", Metric.path_length(new_path))

visualization.draw_path(meshcat, new_path, [9, 10, 11], path="rrt_star_aware", color=Rgba(0, 1, 0, 1))

rrt_star2 = star.RRTStar(planner2, rrt_star_options)
new_path2 = rrt_star2.return_plan()
new_path2 = imacs.UnwrapToContinuousPathSO3(G2, new_path2, 0)
new_traj2 = imacs.SO3PathToDrakeSlerpTraj(Metric2, new_path2, 0)

print("Bsaeline RRT* path length:", Metric2.path_length(new_path2))

visualization.draw_path(meshcat, new_path2, [9, 10, 11], path="rrt_star_unaware", color=Rgba(1, 0, 0, 1))

while True:
    for traj, name in zip([new_traj, new_traj2], ["Symmetry-Aware RRT*", "Symmetry-Unaware RRT*"]):
        time.sleep(3)
        print("%s path" % name)
        dt = traj.end_time() - traj.start_time()
        dt /= 400
        for t in np.linspace(traj.start_time(), traj.end_time(), 400):
            diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(diagram_context)
            time.sleep(dt)