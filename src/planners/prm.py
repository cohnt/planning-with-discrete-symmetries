import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from src.util import repo_dir
import pickle

import src.planners.imacs as imacs

class PRMOptions:
    def __init__(self, neighbor_radius=1e-1, neighbor_k=12, neighbor_mode="k",
                 check_size=1e-2, max_vertices=1e3, scale=True):
        self.neighbor_radius = neighbor_radius
        self.neighbor_k = neighbor_k
        self.neighbor_mode = neighbor_mode # "radius", "k"
        self.check_size = check_size
        self.max_vertices = int(max_vertices)
        self.scale = scale

class PRM:
    def __init__(self, Sampler, Metric, Interpolator, CollisionChecker, options):
        self.Sampler = Sampler
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

    def build(self, nodes_in=None, verbose=False):
        self.graph = nx.DiGraph()

        if nodes_in is None:
            nodes_in = np.zeros((0, self.Sampler.ambient_dim))
        else:
            assert nodes_in.shape[1] == self.Sampler.ambient_dim

        # Gather collision-free samples
        nodes = nodes_in.copy()
        progress_bar = tqdm(total=self.options.max_vertices, desc="Sampling Nodes", disable=not verbose)
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

        dist_mat, targets = self.Metric.pairwise(nodes)
        np.fill_diagonal(dist_mat, np.inf)

        dimension = self.Sampler.ambient_dim

        # Pick edges to check.
        edges_to_try = dict() # Keys will be tuples (i, j), values will be j_rep
        if self.options.neighbor_mode == "radius":
            for i in range(0, len(nodes)):
                card = i + 1
                r = self.options.neighbor_radius
                if self.options.scale:
                    r *= (np.log(card) / card) ** (1/dimension)
                for j in range(0, i - 1):
                    if dist_mat[i,j] <= r:
                        edges_to_try.update({(i, j): targets[i, j]})
        elif self.options.neighbor_mode == "k":
            edge_counts = np.zeros(len(nodes), int)
            for i in range(len(nodes)):
                card = i + 1
                k = self.options.neighbor_k
                if self.options.scale:
                    k *= np.log(card)
                k = int(np.ceil(k))
                j_list = np.argpartition(dist_mat[i], k)[:k]
                for j in j_list:
                    if i > j:
                        edges_to_try.update({(i, j): targets[i, j]})
                    else:
                        edges_to_try.update({(j, i): targets[j, i]})
        else:
            raise NotImplementedError

        num_edges = len(edges_to_try)
        i_in = [i for i, _ in edges_to_try.keys()]
        j_in = [j for _, j in edges_to_try.keys()]
        qj_in = [qj for qj in edges_to_try.values()]
        dist_in = [dist_mat[i,j] for i, j in edges_to_try.keys()]
        self._maybe_connect_parallel(i_in, j_in, qj_in, dist_in)

        if verbose:
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
        connected = []
        for q_new in [start, goal]:
            nearest_idx_sorted, qis = self._order_neighbors(q_new)
            q_new_idx = len(self.graph)
            self.graph.add_node(q_new_idx, q=q_new)
            connected_this = False
            for i in range(len(nearest_idx_sorted)):
                connected_this = connected_this or self._maybe_connect(q_new_idx, nearest_idx_sorted[i], qis[i])
            connected.append(connected_this)

        # Plan
        start_idx = len(self.graph) - 2
        goal_idx = len(self.graph) - 1
        try:
            path = self._path(start_idx, goal_idx)
        except:
            # if self.Sampler.G.order() > 1:
            #     print("Symmetry")
            # else:
            #     print("Baseline")
            # if not connected[0]:
            #     print("Failed to connect start to the graph.")
            # if not connected[1]:
            #     print("Failed to connect goal to the graph.")
            # if connected[0] and connected[1]:
            #     print("Start and goal were only connected to distinct components of the graph.")
            path = []

        # Remove the last two nodes.
        self.graph.remove_node(len(self.graph)-1)
        self.graph.remove_node(len(self.graph)-1)

        return path

    def _order_neighbors(self, q):
        all_qs = self.nodes()
        dists, qis = self.Metric.pairwise(np.array([q]), all_qs)
        dists = dists.reshape(-1)
        qis = qis.reshape(-1, len(q))
        card = len(self.graph)
        dimension = self.Sampler.ambient_dim
        if self.options.neighbor_mode == "radius":
            r = self.options.neighbor_radius
            if self.options.scale:
                r *= (np.log(card) / card) ** (1/dimension)
            num_within_radius = np.sum(dists <= r)
            idxs = np.argsort(dists)[:num_within_radius]
        elif self.options.neighbor_mode == "k":
            k = self.options.neighbor_k
            if self.options.scale:
                k *= np.log(card)
            k = int(np.ceil(k))
            idxs = np.argsort(dists)[:k]
        return idxs, qis[idxs]

    def _maybe_connect_parallel(self, i_list, j_list, qj_list, dist_list):
        edges = [(self.graph.nodes[i]["q"], qj) for i, qj in zip(i_list, qj_list)]
        results = self.CollisionChecker.CheckEdgesCollisionFree(edges)
        for add, i, j, qj, dist in zip(results, i_list, j_list, qj_list, dist_list):
            if add:
                self.graph.add_edge(i, j, weight=dist, qj=qj)
                _, qi = self.Metric(self.graph.nodes[j]["q"], self.graph.nodes[i]["q"])
                self.graph.add_edge(j, i, weight=dist, qj=qi)

        return results

    def _maybe_connect(self, i, j, qj, dist=None):
        q1 = self.graph.nodes[i]["q"]
        q2 = qj
        if isinstance(self.Sampler, imacs.SO2SampleUniform):
            if np.abs(q2[self.Sampler.symmetry_dof_start] - q1[self.Sampler.symmetry_dof_start]) > np.pi / self.Sampler.G.order():
                import pdb
                pdb.set_trace()
        elif isinstance(self.Sampler, imacs.SO3SampleUniform):
            # TODO test
            pass
        if dist is None:
            dist, _ = self.Metric(q1, q2)
        if self.CollisionChecker.CheckEdgeCollisionFreeParallel(q1, q2):
            self.graph.add_edge(i, j, weight=dist, qj=qj)
            _, qi = self.Metric(self.graph.nodes[j]["q"], q1)
            self.graph.add_edge(j, i, weight=dist, qj=qi)
            return True
        else:
            return False

    def _path(self, i, j):
        path_idx = nx.shortest_path(self.graph, source=i, target=j)
        path = []
        for idx in range(len(path_idx) - 1):
            i, j = path_idx[idx], path_idx[idx + 1]
            path.append((self.graph.nodes[i]["q"],
                         self.graph[i][j]["qj"]))
        return path

    def nodes(self):
        all_qs = np.array([self.graph.nodes[i]["q"] for i in range(len(self.graph))])
        return all_qs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["CollisionChecker"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.CollisionChecker = None

    def save(self, fname):
        with open(repo_dir() + "/data/" + fname, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname, CollisionChecker):
        with open(repo_dir() + "/data/" + fname, "rb") as f:
            roadmap = pickle.load(f)
        roadmap.CollisionChecker = CollisionChecker
        return roadmap