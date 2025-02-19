import sys
sys.path.append("..")

import copy
import time
import numpy as np
from tqdm import tqdm
import argparse

import src.asymptotic_optimality_parameters as aop
import src.planners.imacs as imacs
import src.planners.rrt as rrt
import src.planners.star as star
import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d
import src.worlds.path_planning_3d as path_planning_3d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory
)

meshcat = StartMeshcat()

parser = argparse.ArgumentParser(
    prog="statistics_rrt.py",
    description="Measure statistics for the RRT planner in a certain domain.")

parser.add_argument("--shape", type=str, required=True)
# Shape must be one of "polygon", "pyramid", "prism", "tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron"

parser.add_argument("--n_sides", type=int, required=False)
# Only required if shape is polygon, pyramid, or prism

parser.add_argument("--n_worlds", type=int, default=10)
parser.add_argument("--n_pairs_per_world", type=int, default=100)
parser.add_argument("--rrt_nodes_max", type=int, default=1000)

args = parser.parse_args()
if args.shape == "polygon":
    task_space_dimension = 2
    G = symmetry.CyclicGroupSO2(args.n_sides)
    G_name = str(args.n_sides) + "-" + args.shape
elif args.shape == "pyramid":
    task_space_dimension = 3
    G = symmetry.CyclicGroupSO3(args.n_sides)
    G_name = str(args.n_sides) + "-" + args.shape
    dualshape = False # Ignored
elif args.shape == "prism":
    task_space_dimension = 3
    G = symmetry.DihedralGroup(args.n_sides)
    G_name = str(args.n_sides) + "-" + args.shape
    dualshape = False # Ignored
elif args.shape == "tetrahedron":
    task_space_dimension = 3
    G = symmetry.TetrahedralGroup()
    G_name = args.shape
    dualshape = False # Ignored
elif args.shape == "cube":
    task_space_dimension = 3
    G = symmetry.OctahedralGroup()
    G_name = args.shape
    dualshape = False
elif args.shape == "octahedron":
    task_space_dimension = 3
    G = symmetry.OctahedralGroup()
    G_name = args.shape
    dualshape = True
elif args.shape == "dodecahedron":
    task_space_dimension = 3
    G = symmetry.IcosahedralGroup()
    G_name = args.shape
    dualshape = False
elif args.shape == "icosahedron":
    task_space_dimension = 3
    G = symmetry.IcosahedralGroup()
    G_name = args.shape
    dualshape = True

n_worlds = args.n_worlds
n_pairs_per_world = args.n_pairs_per_world

if task_space_dimension == 2:
    rrt_options = rrt.RRTOptions(max_vertices=args.rrt_nodes_max, max_iters=1e4, step_size=5.0, goal_sample_frequency=0.05, stop_at_goal=False)
elif task_space_dimension == 3:
    rrt_options = rrt.RRTOptions(max_vertices=args.rrt_nodes_max, max_iters=1e4, step_size=5.0, goal_sample_frequency=0.05, stop_at_goal=False)

planners_verbose = False

overall_t0 = time.time()

# User receives:
# RRT online runtime improvement
# RRT path length improvement
# RRT* online runtime improvement (even resources)
# RRT* path length improvement (even resources)
# RRT* online runtime improvement (reduced resources)
# RRT* path length improvement (reduced resources)

if task_space_dimension == 2:
    c_space_dimension = 3
    limits = np.array([[0, 20], [0, 20]])
    params = path_planning_2d.SetupParams(G.order(), limits, 200, 1.25, 0)
    c_space_volume = np.prod(limits[:,1] - limits[:,0]) * aop.s1_volume()

    G_unaware = symmetry.CyclicGroupSO2(1)
    Sampler_unaware = imacs.SO2SampleUniform(G_unaware, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
    Metric_unaware = imacs.SO2DistanceSq(G_unaware, 3, 2)
    Interpolator_unaware = imacs.SO2Interpolate(G_unaware, 3, 2)

    Sampler_aware = imacs.SO2SampleUniform(G, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
    Metric_aware = imacs.SO2DistanceSq(G, 3, 2)
    Interpolator_aware = imacs.SO2Interpolate(G, 3, 2)
elif task_space_dimension == 3:
    c_space_dimension = 6
    limits = np.array([[0, 10], [0, 10], [0, 10]])
    params = path_planning_3d.SetupParams(G, dualshape, limits, 150, 0.7, 0)
    c_space_volume = np.prod(limits[:,1] - limits[:,0]) * aop.so3_volume()

    sampler_limits_lower = np.zeros(12)
    sampler_limits_upper = np.zeros(12)
    sampler_limits_lower[-3:] = [limits[0][0], limits[1][0], limits[2][0]]
    sampler_limits_upper[-3:] = [limits[0][1], limits[1][1], limits[2][1]]

    # We don't create the samplers here, because we need to manually specify their random_seed.
    G_unaware = symmetry.CyclicGroupSO3(1)
    Metric_unaware = imacs.SO3DistanceSq(G_unaware, 12, 0)
    Interpolator_unaware = imacs.SO3Interpolate(G_unaware, 12, 0)

    Metric_aware = imacs.SO3DistanceSq(G, 12, 0)
    Interpolator_aware = imacs.SO3Interpolate(G, 12, 0)
else:
    raise NotImplementedError

rrt_star_radius_original = aop.radius_rrt(c_space_dimension, c_space_volume)
rrt_star_radius_quotient = aop.radius_rrt(c_space_dimension, c_space_volume / G.order())
rrt_star_options_even = star.RRTStarOptions(connection_radius=rrt_star_radius_original, mode="radius")
rrt_star_options_uneven = star.RRTStarOptions(connection_radius=rrt_star_radius_quotient, mode="radius")

print("RRT* Radius Ambient", rrt_star_radius_original)
print("RRT* Radius Quotient", rrt_star_radius_quotient)

print("Running comparison for a %s across %d worlds, with %d plans per world" % (G_name, n_worlds, n_pairs_per_world))
print("Group order: %d" % G.order())
# The following lists will be populated by sublists [rrt_unaware, rrt_aware, rrt*_unaware, rrt*_aware_even, rrt*_aware_uneven]
path_lengths = []
runtimes = []
for random_seed in range(n_worlds):
    params.seed = random_seed
    if task_space_dimension == 2:
        diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)
        CollisionCheckerWrapper = CollisionChecker
    else:
        diagram, CollisionChecker = path_planning_3d.build_env(meshcat, params)
        CollisionCheckerWrapper = imacs.SO3CollisionCheckerWrapper(CollisionChecker, 12, 0)

    if task_space_dimension == 3:
        Sampler_unaware = imacs.SO3SampleUniform(G_unaware, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
        Sampler_aware = imacs.SO3SampleUniform(G, 12, 0, sampler_limits_lower, sampler_limits_upper, random_seed=random_seed)
    
    planner_unaware = rrt.RRT(Sampler_unaware, Metric_unaware, Interpolator_unaware, CollisionCheckerWrapper, rrt_options)
    planner_aware = rrt.RRT(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, rrt_options)

    start_goal_pairs = []
    while len(start_goal_pairs) < n_pairs_per_world:
        q0, q1 = Sampler_aware(2)
        if CollisionCheckerWrapper.CheckConfigCollisionFree(q0) and CollisionCheckerWrapper.CheckConfigCollisionFree(q1):
            start_goal_pairs.append((q0, q1))

    for count, (start, goal) in enumerate(tqdm(start_goal_pairs, disable=planners_verbose)):
        path_lengths.append([])
        runtimes.append([])

        # RRT symmetry-unaware and -aware plans
        full_runtimes = []
        if planners_verbose:
            print("World %d, plan %d" % (random_seed, count))
        for planner in [planner_unaware, planner_aware]:
            np.random.seed(random_seed)
            t0 = time.time()
            path, dt = planner.plan(start, goal, verbose=planners_verbose, return_time_to_goal=True)
            t1 = time.time()
            full_runtimes.append(t1 - t0)
            runtimes[-1].append(dt)

            if task_space_dimension == 2:
                path = imacs.UnwrapToContinuousPath2d(planner.Sampler.G, path, planner.Sampler.symmetry_dof_start)
            else:
                path = imacs.UnwrapToContinuousPathSO3(planner.Sampler.G, path, planner.Sampler.symmetry_dof_start)

            if len(path) == 0:
                path_lengths[-1].append(np.inf)
            else:
                path_lengths[-1].append(planner.Metric.path_length(path))

        # RRT* symmetry-unware, -aware (equal resources), and -aware (unequal resources) plans
        symmetry_options = [rrt_star_options_even, rrt_star_options_even, rrt_star_options_uneven]
        planners = [planner_unaware, planner_aware, planner_aware]
        existing_time_indices = [0, 1, 1]
        for rrt_star_options, planner, existing_time_index in zip(symmetry_options, planners, existing_time_indices):
            if path_lengths[-1][existing_time_index] == np.inf:
                runtimes[-1].append(full_runtimes[existing_time_index])
                path_lengths[-1].append(np.inf)
                continue

            t0 = time.time()
            rrt_star = star.RRTStar(copy.deepcopy(planner), rrt_star_options, verbose=planners_verbose)
            path = rrt_star.return_plan()
            t1 = time.time()
            runtimes[-1].append((t1 - t0) + full_runtimes[existing_time_index]) # Cost includes symmetry-unaware RRT time.
            if task_space_dimension == 2:
                path = imacs.UnwrapToContinuousPath2d(planner.Sampler.G, path, planner.Sampler.symmetry_dof_start)
            else:
                path = imacs.UnwrapToContinuousPathSO3(planner.Sampler.G, path, planner.Sampler.symmetry_dof_start)

            if len(path) == 0:
                path_lengths[-1].append(np.inf)
            else:
                path_lengths[-1].append(planner.Metric.path_length(path))

path_lengths = np.asarray(path_lengths)
runtimes = np.asarray(runtimes)

print("Symmetry success rate: %f" % (np.isfinite(path_lengths[:,0]).sum() / path_lengths.shape[0]))
print("Baseline success rate: %f" % (np.isfinite(path_lengths[:,1]).sum() / path_lengths.shape[0]))

mask = np.logical_and(*np.isfinite(path_lengths[:,:2]).T)

def compare(new_idx, old_idx):
    path_lengths_to_compare = path_lengths[mask]
    path_improvement = np.divide(path_lengths_to_compare[:,old_idx], path_lengths_to_compare[:,new_idx]) # Old / New

    runtimes_to_compare = runtimes[mask]
    time_improvement = np.divide(runtimes_to_compare[:,old_idx], runtimes_to_compare[:,new_idx]) # Old / New

    return path_improvement, time_improvement

# RRT comparison
path_improvement, time_improvement = compare(1, 0)
print("\nRRT Comparison")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

# RRT* even comparison
path_improvement, time_improvement = compare(3, 2)
print("\nRRT* Comparison (Equal resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

# RRT* uneven comparison
path_improvement, time_improvement = compare(4, 2)
print("\nRRT* Comparison (Unequal resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

overall_t1 = time.time()
print("Total script runtime", (overall_t1 - overall_t0))