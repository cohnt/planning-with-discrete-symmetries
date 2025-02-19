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
    prog="statistics_rrt.py",
    description="Measure statistics for the RRT planner in a certain domain.")

parser.add_argument("--shape", type=str, required=True)
# Shape must be one of "polygon", "pyramid", "prism", "tetrahedron", "cube", "octahedron", "dodecahedron", "icosahedron"

parser.add_argument("--n_sides", type=int, required=False)
# Only required if shape is polygon, pyramid, or prism

parser.add_argument("--n_worlds", type=int, default=10)
parser.add_argument("--n_pairs_per_world", type=int, default=100)

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

# All parameters besides max_vertices are set later in the script.
prm_options = prm.PRMOptions(max_vertices=1000 * G.order(), neighbor_k=None, neighbor_radius=None, neighbor_mode=None, scale=True, max_ram_pairwise_gb=10)

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

prm_star_radius_original = aop.radius_prm(c_space_dimension, c_space_volume)
prm_star_radius_quotient = aop.radius_prm(c_space_dimension, c_space_volume / G.order())
prm_star_k = aop.knn_prm(c_space_dimension)

prm_options.neighbor_k = prm_star_k
prm_options.neighbor_radius = prm_star_radius_original

r_prm_options = copy.deepcopy(prm_options)
r_prm_options.neighbor_mode = "radius"
k_prm_options = copy.deepcopy(prm_options)
k_prm_options.neighbor_mode = "k"

r_prm_options_uneven = copy.deepcopy(r_prm_options)
r_prm_options_uneven.max_vertices = int(r_prm_options_uneven.max_vertices / G.order())
r_prm_options_uneven.neighbor_radius = prm_star_radius_quotient

k_prm_options_uneven = copy.deepcopy(k_prm_options)
k_prm_options_uneven.max_vertices = int(k_prm_options_uneven.max_vertices / G.order())

print("PRM* Radius Ambient", prm_star_radius_original)
print("PRM* Radius Quotient", prm_star_radius_quotient)
print("PRM* k (Both)", prm_star_k)

print("Running comparison for a %s across %d worlds, with %d plans per world" % (G_name, n_worlds, n_pairs_per_world))
print("Group order: %d" % G.order())
# The following lists will be populated by sublists [radius_prm_unaware_even, radius_prm_aware_even, knn_prm_unaware_even, knn_prm_aware_even,
#                                                    radius_prm_aware_uneven, knn_prm_aware_uneven]
path_lengths = [] # Will be shape (n_worlds * n_pairs, n_planners)
runtimes = [] # Will be shape (n_worlds, n_planners)
for random_seed in tqdm(range(n_worlds), disable=planners_verbose):
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

    prm_r_unaware_even = prm.PRM(Sampler_unaware, Metric_unaware, Interpolator_unaware, CollisionCheckerWrapper, r_prm_options)
    prm_r_aware_even = prm.PRM(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, r_prm_options)
    prm_k_unaware_even = prm.PRM(Sampler_unaware, Metric_unaware, Interpolator_unaware, CollisionCheckerWrapper, k_prm_options)
    prm_k_aware_even = prm.PRM(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, k_prm_options)
    prm_r_aware_uneven = prm.PRM(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, r_prm_options_uneven)
    prm_k_aware_uneven = prm.PRM(Sampler_aware, Metric_aware, Interpolator_aware, CollisionCheckerWrapper, k_prm_options_uneven)

    planners = [prm_r_unaware_even, prm_r_aware_even, prm_k_unaware_even, prm_k_aware_even, prm_r_aware_uneven, prm_k_aware_uneven]

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

print("KNN-PRM symmetry success rate: %f" % (np.isfinite(path_lengths[:,3]).sum() / path_lengths.shape[0]))
print("KNN-PRM symmetry success rate (reduced resources): %f" % (np.isfinite(path_lengths[:,5]).sum() / path_lengths.shape[0]))
print("KNN-PRM baseline success rate: %f" % (np.isfinite(path_lengths[:,2]).sum() / path_lengths.shape[0]))

print("Radius-PRM symmetry success rate: %f" % (np.isfinite(path_lengths[:,1]).sum() / path_lengths.shape[0]))
print("Radius-PRM symmetry success rate (reduced resources): %f" % (np.isfinite(path_lengths[:,4]).sum() / path_lengths.shape[0]))
print("Radius-PRM baseline success rate: %f" % (np.isfinite(path_lengths[:,0]).sum() / path_lengths.shape[0]))

mask_radius_even = np.logical_and(*np.isfinite(path_lengths[:,[0,1]]).T)
mask_radius_uneven = np.logical_and(*np.isfinite(path_lengths[:,[0,4]]).T)
mask_knn_even = np.logical_and(*np.isfinite(path_lengths[:,[2,3]]).T)
mask_knn_uneven = np.logical_and(*np.isfinite(path_lengths[:,[2,5]]).T)

def compare(new_idx, old_idx, mask):
    path_lengths_to_compare = path_lengths[mask]
    path_improvement = np.divide(path_lengths_to_compare[:,old_idx], path_lengths_to_compare[:,new_idx]) # Old / New

    runtimes_to_compare = runtimes.copy()
    time_improvement = np.divide(runtimes_to_compare[:,old_idx], runtimes_to_compare[:,new_idx]) # Old / New

    return path_improvement, time_improvement

# KNN-PRM even comparison
path_improvement, time_improvement = compare(3, 2, mask_knn_even)
print("\nKNN-PRM Comparison (Equal Resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

# Radius-PRM even comparison
path_improvement, time_improvement = compare(1, 0, mask_radius_even)
print("\nRadius-PRM Comparison (Equal Resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

# KNN-PRM unveven comparison
path_improvement, time_improvement = compare(5, 2, mask_knn_uneven)
print("\nKNN-PRM Comparison (Unequal Resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

# Radius-PRM unveven comparison
path_improvement, time_improvement = compare(4, 0, mask_radius_uneven)
print("\nRadius-PRM Comparison (Unequal Resources)")
print("Relative path length decrease factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (path_improvement.mean(), path_improvement.std(), (path_improvement > 1).sum() / len(path_improvement)))
print("Relative runtime speedup factor vs baseline: mean %f ; std %f ; percentage that improved %f" % (time_improvement.mean(), time_improvement.std(), (time_improvement > 1).sum() / len(time_improvement)))

overall_t1 = time.time()
print("Total script runtime", (overall_t1 - overall_t0))