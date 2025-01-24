import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import time

class RRTOptions:
    def __init__(self, step_size=1e0, check_size=1e-2, max_vertices=1e3,
                 max_iters=1e4, goal_sample_frequency=0.05, always_swap=False,
                 timeout=np.inf, stop_at_goal=True):
        self.step_size = step_size
        self.check_size = check_size
        self.max_vertices = int(max_vertices)
        self.max_iters = int(max_iters)
        self.goal_sample_frequency = goal_sample_frequency
        self.always_swap = always_swap
        self.timeout = timeout
        self.stop_at_goal = stop_at_goal
        assert self.goal_sample_frequency >= 0
        assert self.goal_sample_frequency <= 1

class RRT:
    def __init__(self, Sampler, Metric, Interpolator, CollisionChecker, options):
        self.Sampler = Sampler
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

        self.goal_idx = None

    def plan(self, start, goal, verbose=False):
        t0 = time.time()

        self.tree = nx.DiGraph()
        self.tree.add_node(0, q=start)
        success = False

        iters = tqdm(total=self.options.max_iters, position=0, desc="Iterations", disable=not verbose)
        vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices", disable=not verbose)
        for i in range(self.options.max_iters):
            if time.time() - t0 > self.options.timeout:
                break
            iters.update(1)
            old_tree_size = len(self.tree)
            if len(self.tree) >= self.options.max_vertices:
                break
            if success and self.options.stop_at_goal:
                break
            sample_goal = np.random.random() < self.options.goal_sample_frequency and not success
            q_subgoal = goal.copy() if sample_goal else self.Sampler(1).flatten()
            q_near_idx = self._nearest_idx(q_subgoal)
            _, q_subgoal = self.Metric(self.tree.nodes[q_near_idx]["q"], q_subgoal)
            while len(self.tree) < self.options.max_vertices:
                status = self._extend(q_near_idx, q_subgoal)
                q_near_idx = len(self.tree)-1
                if status == "stopped" or status == "reached":
                    break
                if not success:
                    q_new_idx = len(self.tree) - 1
                    q_new = self.tree.nodes[len(self.tree)-1]["q"]
                    dist, qj = self.Metric(q_new, goal)
                    if dist <= self.options.step_size:
                        success = self._maybe_add_and_connect(q_new_idx, qj)
                        if success:
                            self.goal_idx = q_new_idx
                        break
            vertices.update(len(self.tree) - old_tree_size)

        if success:
            return self._path(0, self.goal_idx)
        else:
            return []

    def _nearest_idx(self, q_subgoal):
        all_qs = np.array([self.tree.nodes[i]["q"] for i in range(len(self.tree))])
        dists, _ = self.Metric.pairwise(np.array([q_subgoal]), all_qs)
        return np.argmin(dists)

    def _extend(self, q_near_idx, q_subgoal):
        q_near = self.tree.nodes[q_near_idx]["q"]
        dist_to_subgoal, qj = self.Metric(q_near, q_subgoal)
        trying_to_reach = dist_to_subgoal < self.options.step_size
        if trying_to_reach:
            q_new = q_subgoal
        else:
            step = q_subgoal - q_near
            unit_step = step / self.Metric(q_near, q_subgoal)[0]
            q_new = q_near + self.options.step_size * unit_step

        if self._maybe_add_and_connect(q_near_idx, q_new):
            if trying_to_reach:
                return "reached"
            else:
                return "extended"
        else:
            return "stopped"

    def _maybe_add_and_connect(self, q_near_idx, q_new):
        q_near = self.tree.nodes[q_near_idx]["q"]
        can_add = self.CollisionChecker.CheckEdgeCollisionFreeParallel(q_near, q_new)
        if can_add:
            q_new_idx = len(self.tree)
            self.tree.add_node(q_new_idx, q=q_new)
            self.tree.add_edge(q_near_idx, q_new_idx, weight=self.options.step_size, qj=q_new)
        return can_add

    def _path(self, i, j):
        path_idx = nx.shortest_path(self.tree, source=i, target=j)
        path = []
        for idx in range(len(path_idx) - 1):
            i, j = path_idx[idx], path_idx[idx + 1]
            path.append((self.tree.nodes[i]["q"],
                         self.tree[i][j]["qj"]))
        return path

    def nodes(self):
        all_qs = np.array([self.tree.nodes[i]["q"] for i in range(len(self.tree))])
        return all_qs

class BiRRT:
    def __init__(self, Sampler, Metric, Interpolator, CollisionChecker, options):
        self.Sampler = Sampler
        self.Metric = Metric
        self.Interpolator = Interpolator
        self.CollisionChecker = CollisionChecker
        self.options = options

    def plan(self, start, goal, verbose=False):
        t0 = time.time()

        self.tree_a = nx.DiGraph()
        self.tree_b = nx.DiGraph()

        self.tree_a.add_node(0, q=start)
        self.tree_b.add_node(0, q=goal)

        success = False
        iters = tqdm(total=self.options.max_iters, position=0, desc="Iterations", disable=not verbose)
        vertices = tqdm(total=self.options.max_vertices, position=1, desc="Vertices", disable=not verbose)
        for i in range(self.options.max_iters):
            if time.time() - t0 > self.options.timeout:
                break
            iters.update(1)

            old_tree_size = len(self.tree_a) + len(self.tree_b)
            if old_tree_size >= self.options.max_vertices or success == True:
                break

            q_subgoal = self.Sampler(1).flatten()
            q_near_idx = self._nearest_idx(self.tree_a, q_subgoal)
            _, q_subgoal = self.Metric(self.tree_a.nodes[q_near_idx]["q"], q_subgoal)
            nodes_added = 0
            while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
                status = self._extend(self.tree_a, q_near_idx, q_subgoal)
                if status == "stopped":
                    break
                nodes_added += 1
                if status == "reached":
                    break
                q_near_idx = len(self.tree_a)-1

            if nodes_added == 0:
                if self.options.always_swap:
                    self.tree_a, self.tree_b = self.tree_b, self.tree_a
                continue

            vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
            old_tree_size = len(self.tree_a) + len(self.tree_b)

            selected = np.random.randint(1, nodes_added+1)
            q_subgoal_idx = len(self.tree_a) - selected
            q_subgoal = self.tree_a.nodes[q_subgoal_idx]["q"]
            q_near_idx = self._nearest_idx(self.tree_b, q_subgoal)
            _, q_subgoal = self.Metric(self.tree_b.nodes[q_near_idx]["q"], q_subgoal)

            while len(self.tree_a) + len(self.tree_b) < self.options.max_vertices:
                status = self._extend(self.tree_b, q_near_idx, q_subgoal)
                if status == "stopped":
                    break
                nodes_added += 1
                if status == "reached":
                    success = True
                    break
                q_near_idx = len(self.tree_b)-1

            vertices.update(len(self.tree_a) + len(self.tree_b) - old_tree_size)
            self.tree_a, self.tree_b = self.tree_b, self.tree_a

        if success:
            path_a = self._path(self.tree_a, 0, len(self.tree_a)-1)
            path_b = self._path(self.tree_b, q_subgoal_idx, 0)
            path = path_a + path_b
            if self.Metric(path[0][0], start)[0] > 1e-15:
                path.reverse()
                new_path = [(bar, foo) for foo, bar in path]
                path = new_path
            return path
        else:
            return []

    def _nearest_idx(self, tree, q_subgoal):
        all_qs = np.array([tree.nodes[i]["q"] for i in range(len(tree))])
        dists, _ = self.Metric.pairwise(np.array([q_subgoal]), all_qs)
        return np.argmin(dists)

    def _extend(self, tree, q_near_idx, q_subgoal):
        q_near = tree.nodes[q_near_idx]["q"]
        dist_to_subgoal, qj = self.Metric(q_near, q_subgoal)
        trying_to_reach = dist_to_subgoal < self.options.step_size
        if trying_to_reach:
            q_new = q_subgoal
        else:
            step = q_subgoal - q_near
            unit_step = step / self.Metric(q_near, q_subgoal)[0]
            q_new = q_near + self.options.step_size * unit_step

        if self._maybe_add_and_connect(tree, q_near_idx, q_new):
            if trying_to_reach:
                return "reached"
            else:
                return "extended"
        else:
            return "stopped"

    def _maybe_add_and_connect(self, tree, q_near_idx, q_new):
        q_near = tree.nodes[q_near_idx]["q"]
        can_add = self.CollisionChecker.CheckEdgeCollisionFreeParallel(q_near, q_new)
        if can_add:
            q_new_idx = len(tree)
            tree.add_node(q_new_idx, q=q_new)
            tree.add_edge(q_near_idx, q_new_idx, weight=self.options.step_size, qj=q_new)
            _, qi = self.Metric(tree.nodes[q_new_idx]["q"], tree.nodes[q_near_idx]["q"])
            tree.add_edge(q_new_idx, q_near_idx, weight=self.options.step_size, qj=qi)
        return can_add

    def _path(self, tree, i, j):
        path_idx = nx.shortest_path(tree, source=i, target=j)
        path = []
        for idx in range(len(path_idx) - 1):
            i, j = path_idx[idx], path_idx[idx + 1]
            path.append((tree.nodes[i]["q"],
                         tree[i][j]["qj"]))
        return path