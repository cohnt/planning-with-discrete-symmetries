import sys
sys.path.append("..")

import time
import numpy as np

from tqdm import tqdm

import src.planners.imacs as imacs
import src.planners.rrt as rrt
import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory
)

sides_to_try = [3]
n_worlds = 5
n_pairs_per_world = 30

meshcat = StartMeshcat()
options = rrt.RRTOptions(max_vertices=1e3, max_iters=1e4, goal_sample_frequency=0.05)

for n_sides in sides_to_try:
    print("Running comparison for an %d-gon across %d worlds, with %d plans per world" % (n_sides, n_worlds, n_pairs_per_world))
    path_lengths = []
    node_counts = []
    for obstacle_seed in range(n_worlds):
        limits = [[0, 20], [0, 20]]
        params = path_planning_2d.SetupParams(n_sides, limits, 200, 1.25, obstacle_seed)
        diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

        G1 = symmetry.CyclicGroupSO2(n_sides)
        Sampler1 = imacs.SO2SampleUniform(G1, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
        Metric1 = imacs.SO2DistanceSq(G1, 3, 2)
        Interpolator1 = imacs.SO2Interpolate(G1, 3, 2)
        planner1 = rrt.BiRRT(Sampler1, Metric1, Interpolator1, CollisionChecker, options)

        G2 = symmetry.CyclicGroupSO2(1)
        Sampler2 = imacs.SO2SampleUniform(G2, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
        Metric2 = imacs.SO2DistanceSq(G2, 3, 2)
        Interpolator2 = imacs.SO2Interpolate(G2, 3, 2)
        planner2 = rrt.BiRRT(Sampler2, Metric2, Interpolator2, CollisionChecker, options)

        start_goal_pairs = []
        while len(start_goal_pairs) < n_pairs_per_world:
            q0, q1 = Sampler1(2)
            if CollisionChecker.CheckConfigCollisionFree(q0) and CollisionChecker.CheckConfigCollisionFree(q1):
                start_goal_pairs.append((q0, q1))

        for start, goal in tqdm(start_goal_pairs):
            path_lengths.append([])
            node_counts.append([])
            for planner in [planner1, planner2]:
                np.random.seed(0)
                path = planner.plan(start, goal)
                path = imacs.UnwrapToContinuousPath2d(planner.Sampler.G, path, planner.Sampler.symmetry_dof_start)

                node_counts[-1].append(len(planner.tree_a) + len(planner.tree_b))

                if len(path) == 0:
                    path_lengths[-1].append(np.inf)
                    continue

                path_lengths[-1].append(planner.Metric.path_length(path))

    node_counts = np.asarray(node_counts)
    path_lengths = np.asarray(path_lengths)

    print("Symmetry success rate: %f" % (np.isfinite(path_lengths[:,0]).sum() / path_lengths.shape[0]))
    print("Baseline success rate: %f" % (np.isfinite(path_lengths[:,1]).sum() / path_lengths.shape[0]))

    mask = np.logical_and(*np.isfinite(path_lengths).T)
    path_lengths_to_compare = path_lengths[mask]
    relative_improvement = np.divide(path_lengths_to_compare[:,0], path_lengths_to_compare[:,1]) # New / Old

    node_counts_to_compare = node_counts[mask]
    node_relative_improvement = np.divide(node_counts_to_compare[:,0], node_counts_to_compare[:,1]) # New / Old


    print("Relative path length of the symmetry-aware path compared to the baseline: mean %f, std %f, percentage that improved %f" % (relative_improvement.mean(), relative_improvement.std(), (relative_improvement < 1).sum() / len(relative_improvement)))
    print("Relative number of nodes in the symmetry-aware path compared to the baseline: mean %f, std %f, percentage that improved %f" % (node_relative_improvement.mean(), node_relative_improvement.std(), (node_relative_improvement < 1).sum() / len(node_relative_improvement)))
    print("\n")