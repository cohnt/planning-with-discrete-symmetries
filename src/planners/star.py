import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import time

import src.planners.rrt as rrt

class RRTStarOptions:
    def __init__(self, connection_radius=5.0, connection_k=12, mode="radius"):
        self.connection_radius = connection_radius
        self.connection_k = connection_k
        self.mode = mode

        assert isinstance(self.connection_k, int)
        assert self.connection_radius > 0
        assert self.connection_k > 0
        assert self.mode in ["radius", "k"]

class RRTStar:
    def __init__(self, existing_rrt, options, verbose=False):
        self.rrt = existing_rrt
        self.options = options

        self._rewire_tree(verbose)

    def _rewire_tree(self, verbose):
        nodes = self.rrt.nodes()
        dist_sq_mat, targets = self.rrt.Metric.pairwise(nodes)
        dist_mat = np.sqrt(dist_sq_mat)

        # Compute cost-to-come
        self._compute_cost_to_come()

        overall_start = time.time()
        recomputation_time = 0

        # Incrementally rewire
        for j in tqdm(range(1, len(self.rrt.tree)), desc="RRT* Rewiring", disable=not verbose):
            # Find neighbors
            candidate_dists = dist_mat[j,:j]
            if self.options.mode == "k":
                partitioned_indices = np.argpartition(candidate_dists, self.options.connection_k)
                candidate_nodes = partitioned_indices[:self.options.connection_k]
            elif self.options.mode == "radius":
                candidate_nodes = np.where(candidate_dists <= self.options.connection_radius)[0]
            else:
                raise(NotImplementedError)

            # Record current best
            in_edges = self.rrt.tree.in_edges(j)
            assert len(in_edges) == 1
            best_ij = list(in_edges)[0]
            best_i = best_ij[0]
            best_cost = self.rrt.tree.nodes[best_i]["cost to come"] + dist_mat[best_i, j]

            # Determine candidate edges
            candidate_costs = np.array([self.rrt.tree.nodes[i]["cost to come"] + dist_mat[i,j] for i in range(j)])
            mask = candidate_costs < best_cost
            candidate_nodes = candidate_nodes[mask[candidate_nodes]]
            candidate_nodes_sorted = candidate_nodes[np.argsort(candidate_costs[candidate_nodes])]

            # Check candidate edges for collision (in order)
            for candidate_node in candidate_nodes_sorted:
                qi = self.rrt.tree.nodes[candidate_node]["q"]
                qj = targets[candidate_node, j]
                can_rewire = self.rrt.CollisionChecker.CheckEdgeCollisionFreeParallel(qi, qj)
                if can_rewire:
                    self.rrt.tree.remove_edge(best_i, j)
                    self.rrt.tree.add_edge(candidate_node, j, weight=dist_mat[candidate_node, j], qj=qj)

                    # We need to recompute cost to come. This could be made more efficient
                    t0 = time.time()
                    self._compute_cost_to_come()
                    t1 = time.time()
                    recomputation_time += t1 - t0

                    # Since we went in order, we can break
                    break

        overall_time = time.time() - overall_start
        if verbose:
            print("Overall runtime", overall_time)
            print("Recomputation runtime", recomputation_time)

    def _compute_cost_to_come(self):
        path_lengths = nx.shortest_path_length(self.rrt.tree, source=0, weight="weight")
        for i, data in self.rrt.tree.nodes(data=True):
            data["cost to come"] = path_lengths[i]

    def return_plan(self):
        return self.rrt._path(0, self.rrt.goal_idx)