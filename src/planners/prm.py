import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from src.util import repo_dir
import pickle
import time
import gc

import src.planners.imacs as imacs

class PRMOptions:
    def __init__(self, neighbor_radius=1e-1, neighbor_k=12, neighbor_mode="k",
                 check_size=1e-2, max_vertices=1e3, scale=True, max_ram_pairwise_gb=10, min_k=1):
        self.neighbor_radius = neighbor_radius
        self.neighbor_k = neighbor_k
        self.neighbor_mode = neighbor_mode # "radius", "k"
        self.check_size = check_size
        self.max_vertices = int(max_vertices)
        self.scale = scale
        self.max_ram_pairwise_gb = max_ram_pairwise_gb
        self.min_k = min_k

class PRM:
    def __init__(self, Sampler, Metric, Interpolator, CollisionChecker, options):
        self.Sampler = Sampler
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

        # Compute anticipated memory usage of pairwise distance computations
        expected_bytes1 = 1 # Number of bytes per sample-squared from the distance matrix
        expected_bytes1 *= self.Sampler.G.order()
        expected_bytes1 *= 3 ** 2 # 3x3 matrices for SO(3)
        expected_bytes1 *= 8 # Numpy uses doubles, which need 8 bytes

        expected_bytes2 = 1 # Number of bytes per sample-squared from the targets matrix
        expected_bytes2 *= self.Sampler.ambient_dim
        expected_bytes2 *= 8

        expected_bytes = expected_bytes1 + expected_bytes2
        expected_bytes *= 2 # Empirically, the overhead works out to about twice the memory needed to store the output
        max_ram = self.options.max_ram_pairwise_gb * (10 ** 9) # Allow using up to 50GB of RAM for pairwise distance computations.

        # samples^2 * mult = max_ram -> samples = sqrt(max_ram / mult)
        self.pairwise_max_block_size = int(np.sqrt(max_ram / expected_bytes))

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

        # Compute pairwise distances.
        dist_mat = np.zeros((len(nodes), len(nodes)))

        targets_too_large = False
        try:
            targets = np.zeros((len(nodes), len(nodes), self.Sampler.ambient_dim))
        except:
            if verbose:
                print("Warning: cannot precompute all edge targets due to memory limitations.")
            targets = None
            targets_too_large = True

        n_blocks = (int(len(nodes) / self.pairwise_max_block_size) + 1) ** 2
        if verbose:
            if n_blocks == 1:
                print("Computing pairwise distances in a single block")
            else:
                print("Using %dx%d blocks for pairwise distance computations" % (self.pairwise_max_block_size, self.pairwise_max_block_size))
        progress_bar = tqdm(total=n_blocks, desc="Pairwise Distance Blocks", disable=not verbose)
        for i in range(0, len(nodes), self.pairwise_max_block_size):
            i_max = i + self.pairwise_max_block_size
            for j in range(0, len(nodes), self.pairwise_max_block_size):
                progress_bar.update(1)
                # TODO: Maybe make this i to len(nodes)?
                if i == j:
                    block_dist, block_targets = self.Metric.pairwise(nodes[i:i_max])
                    dist_mat[i:i_max, i:i_max] = block_dist
                    if not targets_too_large:
                        targets[i:i_max, i:i_max] = block_targets
                    del block_dist, block_targets
                    gc.collect()
                else:
                    j_max = j + self.pairwise_max_block_size
                    block_dist, block_targets = self.Metric.pairwise(nodes[i:i_max], nodes[j:j_max])
                    dist_mat[i:i_max, j:j_max] = block_dist
                    if not targets_too_large:
                        targets[i:i_max, j:j_max] = block_targets
                    del block_dist, block_targets
                    gc.collect()
        progress_bar.close()

        np.fill_diagonal(dist_mat, np.inf)

        dimension = self.Sampler.ambient_dim

        # Pick edges to check.
        edges_to_try = dict() # Keys will be tuples (i, j), values will be j_rep
        if self.options.neighbor_mode == "radius":
            progress_bar = tqdm(total=int(0.5 * len(nodes) * (len(nodes) - 1)), desc="Finding Radius-Nearest Neighbors", disable=not verbose)
            for i in range(0, len(nodes)):
                card = i + 1
                r = self.options.neighbor_radius
                if self.options.scale:
                    r *= (np.log(card) / card) ** (1/dimension)
                for j in range(0, i - 1):
                    progress_bar.update(1)
                    if dist_mat[i,j] <= r:
                        target = self.Metric(nodes[i], nodes[j])[1] if targets_too_large else targets[i, j]
                        edges_to_try.update({(i, j): target})
            progress_bar.close()
        elif self.options.neighbor_mode == "k":
            edge_counts = np.zeros(len(nodes), int)
            progress_bar = tqdm(total=len(nodes), desc="Finding k-Nearest Neighbors", disable=not verbose)
            for i in range(len(nodes)):
                progress_bar.update(1)
                card = i + 1
                k = self.options.neighbor_k
                if self.options.scale:
                    k *= np.log(card)
                k = int(np.ceil(k))
                if k < self.options.min_k:
                    k = self.options.min_k
                j_list = np.argpartition(dist_mat[i], k)[:k]
                for j in j_list:
                    if i > j:
                        target = self.Metric(nodes[i], nodes[j])[1] if targets_too_large else targets[i, j]
                        edges_to_try.update({(i, j): target})
                    else:
                        target = self.Metric(nodes[j], nodes[i])[1] if targets_too_large else targets[j, i]
                        edges_to_try.update({(j, i): target})
            progress_bar.close()
        else:
            raise NotImplementedError

        del targets
        gc.collect()

        if verbose:
            print("Checking %d edges for collisions" % len(edges_to_try))
        i_in = [i for i, _ in edges_to_try.keys()]
        j_in = [j for _, j in edges_to_try.keys()]
        qj_in = [qj for qj in edges_to_try.values()]
        dist_in = [dist_mat[i,j] for i, j in edges_to_try.keys()]

        del dist_mat
        gc.collect()

        t0 = time.time()
        self._maybe_connect_parallel(i_in, j_in, qj_in, dist_in)
        t1 = time.time()
        if verbose:
            print("Edges checked in %f seconds (about %d edges per second)." % ((t1 - t0), int(len(edges_to_try) / (t1 - t0))))

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
        edges = np.array([(self.graph.nodes[i]["q"], qj) for i, qj in zip(i_list, qj_list)])
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
        if isinstance(self.Sampler, imacs.SO2SampleUniform) and not isinstance(self.Sampler.symmetry_dof_start, list):
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