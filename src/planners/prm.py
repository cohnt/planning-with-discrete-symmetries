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
        self.neighbor_mode = neighbor_mode # "radius", "k"
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
        self.graph = nx.DiGraph()

        # Gather collision-free samples
        nodes = np.zeros((0, self.Sampler.ambient_dim))
        progress_bar = tqdm(total=self.options.max_vertices, desc="Sampling Nodes")
        while len(nodes) < self.options.max_vertices:
            candidate_nodes = self.Sampler(self.options.max_vertices - len(nodes))
            validity_mask = self.CollisionChecker.CheckConfigsCollisionFree(candidate_nodes)
            nodes = np.append(nodes, candidate_nodes[np.nonzero(validity_mask)], axis=0)
            progress_bar.n = nodes.shape[0]
            progress_bar.refresh()
        progress_bar.close()

        for i in range(len(nodes)):
            self.graph.add_node(i, q=nodes[i])

        # Compute pairwise distances. TODO: factor out?
        dist_mat = np.zeros((len(nodes), len(nodes)))
        targets = np.zeros((len(nodes), len(nodes), self.Sampler.ambient_dim))
        total = int(len(nodes) * (len(nodes) - 1) / 2)
        progress_bar = tqdm(total=total, desc="Computing Distances")
        for i in range(0, len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist_mat[i,j], targets[i,j] = self.Metric(nodes[i], nodes[j])
                dist_mat[j,i] = dist_mat[i,j]
                targets[j,i] = np.full(self.Sampler.ambient_dim, np.nan)
                progress_bar.n += 1
                progress_bar.refresh()
        np.fill_diagonal(dist_mat, np.inf)
        progress_bar.close()

        # Pick edges to check.
        edges_to_try = dict() # Keys will be tuples (i, j), values will be j_rep
        if self.options.neighbor_mode == "radius":
            for i in range(0, len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if dist_mat[i,j] <= self.options.neighbor_radius:
                        edges_to_try.update({(i, j): targets[i, j]})
        elif self.options.neighbor_mode == "k":
            edge_counts = np.zeros(len(nodes), int)
            for i in range(len(nodes)):
                j_list = np.argpartition(dist_mat[i], self.options.neighbor_k)[:self.options.neighbor_k]
                for j in j_list:
                    if i < j:
                        edges_to_try.update({(i, j): targets[i, j]})
                    else:
                        edges_to_try.update({(j, i): targets[j, i]})
        else:
            raise NotImplementedError

        num_edges = len(edges_to_try)
        for (i, j), qj in tqdm(edges_to_try.items(), "Checking Edges for Collisions"):
            assert i < j
            self._maybe_connect(i, j, qj, dist=dist_mat[i,j])

        print("Created a roadmap with %d vertices, %d edges, and %d"
              " connected components." % (len(self.graph),
                                         self.graph.number_of_edges(),
                                         len(list(nx.connected_components(self.graph.to_undirected())))))
        count = 3
        print("%d largest components: %s" % (
            count,
            [len(c) for c in sorted(nx.connected_components(self.graph.to_undirected()), key=len, reverse=True)][:count])
        )

    def plan(self, start, goal):
        # Try to connect the start and goal
        for q_new in [start, goal]:
            nearest_idx_sorted, qis = self._order_neighbors(q_new)
            q_new_idx = len(self.graph)
            self.graph.add_node(q_new_idx, q=q_new)
            for i in range(len(nearest_idx_sorted)):
                self._maybe_connect(q_new_idx, nearest_idx_sorted[i], qis[i])

        # Plan
        start_idx = len(self.graph) - 2
        goal_idx = len(self.graph) - 1
        try:
            path = self._path(start_idx, goal_idx)
        except:
            path = []

        # Remove the last two nodes.
        self.graph.remove_node(len(self.graph)-1)
        self.graph.remove_node(len(self.graph)-1)

        return path

    def _order_neighbors(self, q):
        pairs = [self.Metric(q, self.graph.nodes[i]["q"])
            for i in range(len(self.graph))]
        dists = np.array([foo for foo, _ in pairs])
        qis = [bar for _, bar in pairs]
        if self.options.neighbor_mode == "radius":
            num_within_radius = np.sum(dists <= self.options.neighbor_radius)
            idxs = np.argsort(dists)[:num_within_radius]
        elif self.options.neighbor_mode == "k":
            idxs = np.argsort(dists)[:self.options.neighbor_k]
        return idxs, np.array(qis)[idxs]

    def _maybe_connect(self, i, j, qj, dist=None):
        q1 = self.graph.nodes[i]["q"]
        q2 = qj
        if dist is None:
            dist, _ = self.Metric(q1, q2)
        if self.CollisionChecker.CheckEdgeCollisionFreeParallel(q1, q2):
            self.graph.add_edge(i, j, weight=dist, qj=qj)
            _, qi = self.Metric(self.graph.nodes[j]["q"], q1)
            self.graph.add_edge(j, i, weight=dist, qj=qi)

    def _path(self, i, j):
        path_idx = nx.shortest_path(self.graph, source=i, target=j)
        path = []
        for idx in range(len(path_idx) - 1):
            i, j = path_idx[idx], path_idx[idx + 1]
            path.append(self.graph.nodes[i]["q"])
            path.append(self.graph[i][j]["qj"])
        return path

    # def save(self, fname):
    #     nodes = [self.graph.nodes[i]["q"] for i in range(len(self.graph))]
    #     adj_mat = nx.adjacency_matrix(self.graph)
    #     with open(repo_dir() + "/data/" + fname, "wb") as f:
    #         pickle.dump((nodes, adj_mat), f)
    #         f.close()

    # def load(self, fname):
    #     with open(repo_dir() + "/data/" + fname, "rb") as f:
    #         nodes, adj_mat = pickle.load(f)
    #         self.graph = nx.Graph()
    #         for i, node in enumerate(nodes):
    #             self.graph.add_node(i, q=node)
    #         i_list, j_list = adj_mat.nonzero()
    #         for i, j in zip(i_list, j_list):
    #             if i < j:
    #                 self.graph.add_edge(i, j, weight=self.Distance(nodes[i], nodes[j]))