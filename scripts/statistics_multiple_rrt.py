import sys
sys.path.append(".")
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

parser.add_argument("--dimension", type=int, required=True)
parser.add_argument("--n_sides", type=int, required=False)
parser.add_argument("--n_worlds", type=int, default=10)
parser.add_argument("--n_pairs_per_world", type=int, default=100)
parser.add_argument("--rrt_nodes_max", type=int, default=1000)
parser.add_argument("--step_size", type=float, default=1.0)

args = parser.parse_args()

task_space_dimension = 2
n_copies = int(np.ceil(args.dimension / 3))
cspace_dim = 3 * n_copies

G = symmetry.CyclicGroupSO2(args.n_sides)
G_name = "%d-gon, %d copies, dimension %d" % (args.n_sides, n_copies, args.dimension)

n_worlds = args.n_worlds
n_pairs_per_world = args.n_pairs_per_world

rrt_options = rrt.RRTOptions(max_vertices=args.rrt_nodes_max, max_iters=1e5, step_size=args.step_size, goal_sample_frequency=0.01, stop_at_goal=True)

planners_verbose = True

overall_t0 = time.time()

# User receives:
# RRT online runtime improvement
# RRT path length improvement
# RRT* online runtime improvement (even resources)
# RRT* path length improvement (even resources)
# RRT* online runtime improvement (reduced resources)
# RRT* path length improvement (reduced resources)

c_space_dimension = 3
symmetry_indices = list(range(2, 3*n_copies, 3))
limits = np.array([[0, 20], [0, 20]])
params = path_planning_2d.SetupParams(G.order(), limits, 120, 0.75, 0)
c_space_volume = np.prod(limits[:,1] - limits[:,0]) * aop.s1_volume()

limits_lower = [limits[0][0], limits[1][0], 0] * n_copies
limits_upper = [limits[0][1], limits[1][1], 0] * n_copies
num_to_clip = 3 * n_copies - args.dimension
for i in range(0, num_to_clip):
    clip_idx = -2 - i
    limits_lower[clip_idx] = 10
    limits_upper[clip_idx] = 10

G_unaware = symmetry.CyclicGroupSO2(1)
Sampler_unaware = imacs.SO2SampleUniform(G_unaware, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric_unaware = imacs.SO2DistanceSqMultiple(G_unaware, cspace_dim, symmetry_indices)
Interpolator_unaware = imacs.SO2InterpolateMultiple(G_unaware, cspace_dim, symmetry_indices)

Sampler_aware = imacs.SO2SampleUniform(G, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric_aware = imacs.SO2DistanceSqMultiple(G, cspace_dim, symmetry_indices)
Interpolator_aware = imacs.SO2InterpolateMultiple(G, cspace_dim, symmetry_indices)

print("Running comparison for a %s across %d worlds, with %d plans per world" % (G_name, n_worlds, n_pairs_per_world))
print("Group order: %d" % G.order())
print("Step size: %d" % rrt_options.step_size)
print("Max vertices: %d" % rrt_options.max_vertices)
# The following lists will be populated by sublists [rrt_unaware, rrt_aware]
path_lengths = []
runtimes = []
for random_seed in range(n_worlds):
    params.seed = random_seed
    
    diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params, n_copies=n_copies)
    CollisionCheckerWrapper = CollisionChecker
    
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

            path = imacs.UnwrapToContinuousPath2dMultiple(planner.Sampler.G, path, symmetry_indices)

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

overall_t1 = time.time()
print("Total script runtime", (overall_t1 - overall_t0))