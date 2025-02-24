import sys
sys.path.append("..")

import copy
import time
import numpy as np
from tqdm import tqdm
import argparse

import src.asymptotic_optimality_parameters as aop
import src.planners.imacs as imacs
import src.planners.prm as prm
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
    prog="statistics_multiple_prm.py",
    description="Measure statistics for the PRM planner in the multiple-object context.")

parser.add_argument("--dimension", type=int, required=True)
parser.add_argument("--n_sides", type=int, required=True)
parser.add_argument("--n_worlds", type=int, default=10)
parser.add_argument("--n_pairs_per_world", type=int, default=100)
parser.add_argument("--n_vertices", type=int, default=1000)

args = parser.parse_args()

task_space_dimension = 2
n_copies = int(np.ceil(args.dimension / 3))
cspace_dim = 3 * n_copies

G = symmetry.CyclicGroupSO2(args.n_sides)
G_name = "%d-gon, %d copies, dimension %d" % (args.n_sides, n_copies, args.dimension)

n_worlds = args.n_worlds
n_pairs_per_world = args.n_pairs_per_world

# All parameters besides max_vertices are set later in the script.
prm_options = prm.PRMOptions(max_vertices=args.n_vertices, neighbor_k=100, neighbor_radius=None, neighbor_mode="k", scale=False, max_ram_pairwise_gb=30)

planners_verbose = False

overall_t0 = time.time()

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

prm_star_k = aop.knn_prm(c_space_dimension)

prm_options.neighbor_k = prm_star_k

k_prm_options = copy.deepcopy(prm_options)
k_prm_options.neighbor_mode = "k"

print("PRM* k (Both)", prm_star_k)

print("Running comparison for a %s across %d worlds, with %d plans per world" % (G_name, n_worlds, n_pairs_per_world))
print("Group order: %d" % G.order())
# The following lists will be populated by sublists [radius_prm_unaware_even, radius_prm_aware_even, knn_prm_unaware_even, knn_prm_aware_even,
#                                                    radius_prm_aware_uneven, knn_prm_aware_uneven]
path_lengths = [] # Will be shape (n_worlds * n_pairs, n_planners)
runtimes = [] # Will be shape (n_worlds, n_planners)
for random_seed in range(n_worlds):
    params.seed = random_seed
    diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params, n_copies=n_copies)
    CollisionCheckerWrapper = CollisionChecker

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)

    prm_k_aware_even = prm.PRM(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, k_prm_options)
    prm_k_unaware_even = prm.PRM(Sampler_unaware, Metric_unaware, Interpolator_unaware, CollisionCheckerWrapper, k_prm_options)

    planners = [prm_k_aware_even, prm_k_unaware_even]

    runtimes.append([])
    for planner in planners:
        np.random.seed(random_seed)
        t0 = time.time()
        planner.build(verbose=planners_verbose)
        t1 = time.time()
        runtimes[-1].append(t1 - t0)

    start_goal_pairs = []
    while len(start_goal_pairs) < n_pairs_per_world:
        q0, q1 = Sampler_aware(2)
        if CollisionCheckerWrapper.CheckConfigCollisionFree(q0) and CollisionCheckerWrapper.CheckConfigCollisionFree(q1):
            start_goal_pairs.append((q0, q1))

    for count, (start, goal) in enumerate(tqdm(start_goal_pairs, disable=planners_verbose)):
        path_lengths.append([])

        if planners_verbose:
            print("World %d, plan %d" % (random_seed, count))
        for planner in planners:
            path = planner.plan(start, goal)
            path = imacs.UnwrapToContinuousPath2dMultiple(planner.Sampler.G, path, symmetry_indices)

            if len(path) == 0:
                path_lengths[-1].append(np.inf)
            else:
                path_lengths[-1].append(planner.Metric.path_length(path))

path_lengths = np.asarray(path_lengths)
runtimes = np.asarray(runtimes)

print("KNN-PRM symmetry success rate: %f" % (np.isfinite(path_lengths[:,0]).sum() / path_lengths.shape[0]))
print("KNN-PRM baseline success rate: %f" % (np.isfinite(path_lengths[:,1]).sum() / path_lengths.shape[0]))

mask_knn_even = np.logical_and(*np.isfinite(path_lengths[:,[0,1]]).T)

def compare(new_idx, old_idx, mask):
    path_lengths_to_compare = path_lengths[mask]
    path_improvement = np.divide(path_lengths_to_compare[:,old_idx], path_lengths_to_compare[:,new_idx]) # Old / New

    runtimes_to_compare = runtimes.copy()
    time_improvement = np.divide(runtimes_to_compare[:,old_idx], runtimes_to_compare[:,new_idx]) # Old / New

    return path_improvement, time_improvement

# KNN-PRM even comparison
path_improvement, time_improvement = compare(0, 1, mask_knn_even)
print("\nKNN-PRM Comparison (Equal Resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

overall_t1 = time.time()
print("Total script runtime", (overall_t1 - overall_t0))