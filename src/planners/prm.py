import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from src.util import repo_dir
import pickle
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
import time

class PRMOptions:
    def __init__(self, neighbor_radius=1e-1, neighbor_k=12, neighbor_mode="k",
                 check_size=1e-2, max_vertices=1e3):
        self.neighbor_radius = neighbor_radius
        self.neighbor_k = neighbor_k
        self.neighbor_mode = neighbor_mode # "radius", "k", "min", or "max"
        self.check_size = check_size
        self.max_vertices = int(max_vertices)

class PRM:
    def __init__(self, Sampler, Metric, Interpolator, CollisionChecker, options):
        self.Sampler = Sampler
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

    def build(self):
        self.graph = nx.Graph()

        # Gather collision-free samples
        nodes = np.zeros((0, self.Sampler.ambient_dim))
        progress_bar = tqdm(total = self.options.max_vertices, desc="Sampling Nodes")
        while len(nodes) < self.options.max_vertices:
            candidate_nodes = self.Sampler(self.options.max_vertices - len(nodes))
            validity_mask = self.CollisionChecker.CheckConfigsCollisionFree(candidate_nodes)
            nodes = np.append(nodes, candidate_nodes[validity_mask])
            progress_bar.n = nodes.shape[0]
            progress_bar.refresh()

        # for i in tqdm(range(len(self.graph), self.options.max_vertices)):
        #     while True:
        #         q_new = self.RandomConfig()
        #         if self.ValidityChecker(q_new):
        #             break
        #     nearest_idx_sorted = self._order_neighbors(q_new)
        #     q_new_idx = len(self.graph)
        #     self.graph.add_node(q_new_idx, q=q_new)
        #     for idx in nearest_idx_sorted:
        #         self._maybe_connect(q_new_idx, idx)

        # print("Created a roadmap with %d vertices, %d edges, and %d"
        #       " connected components." % (len(self.graph),
        #                                  self.graph.number_of_edges(),
        #                                  len(list(nx.connected_components(self.graph)))))
        # count = 3
        # print("%d largest components: %s" % (
        #     count,
        #     [len(c) for c in sorted(nx.connected_components(self.graph), key=len, reverse=True)][:count])
        # )

    def plan(self, start, goal, options=None):
        if options is not None:
            self.options = options

        # Check if start and goal are already in the graph
        start_idx = -1
        goal_idx = -1
        for i in range(len(self.graph)):
            if np.linalg.norm(start - self.graph.nodes[i]["q"]) < self.options.check_size:
                start_idx = i
            if np.linalg.norm(goal - self.graph.nodes[i]["q"]) < self.options.check_size:
                goal_idx = i

        if start_idx != -1 and goal_idx != -1:
            # We already have the start and goal in the graph!
            return self._path(start_idx, goal_idx)

        # Try to connect the start and goal
        for q_new in [start, goal]:
            nearest_idx_sorted = self._order_neighbors(q_new)
            q_new_idx = len(self.graph)
            self.graph.add_node(q_new_idx, q=q_new)
            for idx in nearest_idx_sorted:
                self._maybe_connect(q_new_idx, idx)

        # Plan
        start_idx = len(self.graph) - 2
        goal_idx = len(self.graph) - 1
        return self._path(start_idx, goal_idx)

    def _order_neighbors(self, q):
        dists = np.array([self.Distance(q, self.graph.nodes[i]["q"])
            for i in range(len(self.graph))])
        if self.options.neighbor_mode == "radius":
            num_within_radius = np.sum(dists <= self.options.neighbor_radius)
            return np.argsort(dists)[:num_within_radius]
        elif self.options.neighbor_mode == "k":
            return np.argsort(dists)[:self.options.neighbor_k]
        elif self.options.neighbor_mode == "min":
            num_within_radius = np.sum(dists <= self.options.neighbor_radius)
            return np.argsort(dists)[:min(self.options.neighbor_k, num_within_radius)]
        elif self.options.neighbor_mode == "max":
            num_within_radius = np.sum(dists <= self.options.neighbor_radius)
            return np.argsort(dists)[:max(self.options.neighbor_k, num_within_radius)]

    def _maybe_connect(self, i, j):
        q1 = self.graph.nodes[i]["q"]
        q2 = self.graph.nodes[j]["q"]
        step = q2 - q1
        dist = self.Distance(q1, q2)
        unit_step = step / dist
        validity_step = self.options.check_size * unit_step
        step_counts = np.arange(1, 1+int(dist / self.options.check_size))
        np.random.shuffle(step_counts)
        for step_count in step_counts:
            if not self.ValidityChecker(q1 + validity_step * step_count):
                return
        self.graph.add_edge(i, j, weight=dist)

    def _path(self, i, j):
        path_idx = nx.shortest_path(self.graph, source=i, target=j)
        path = [self.graph.nodes[idx]["q"] for idx in path_idx]
        return path

    def save(self, fname):
        nodes = [self.graph.nodes[i]["q"] for i in range(len(self.graph))]
        adj_mat = nx.adjacency_matrix(self.graph)
        with open(repo_dir() + "/data/" + fname, "wb") as f:
            pickle.dump((nodes, adj_mat), f)
            f.close()

    def load(self, fname):
        with open(repo_dir() + "/data/" + fname, "rb") as f:
            nodes, adj_mat = pickle.load(f)
            self.graph = nx.Graph()
            for i, node in enumerate(nodes):
                self.graph.add_node(i, q=node)
            i_list, j_list = adj_mat.nonzero()
            for i, j in zip(i_list, j_list):
                if i < j:
                    self.graph.add_edge(i, j, weight=self.Distance(nodes[i], nodes[j]))