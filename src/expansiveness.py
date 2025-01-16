import numpy as np
import networkx as nx
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class Expansiveness:
    def __init__(self, roadmap):
        print("Processing roadmap to compute expansiveness...")
        # Call with a roadmap, stored as an undirected or directed graph
        self.G = roadmap
        if self.G.is_directed():
            self.G = self.G.to_undirected()

        self.N = len(self.G)

        self.connected_components = list(nx.connected_components(self.G))
        self.component_indices = [-1 for _ in range(self.N)]
        for idx, component in tqdm(enumerate(self.connected_components)):
            for node in component:
                self.component_indices[node] = idx

        self.visible_sets = [set(self.G.neighbors(node)) | {node} for node in self.G.nodes]
        for i in range(self.N):
            assert i in self.visible_sets[i]

        self.visible_lists = [list(s) for s in self.visible_sets]
        self.beta_vws = [[] for _ in range(self.N)]
        for i in tqdm(range(self.N)):
            for j in range(len(self.visible_lists[i])):
                vertex_index = self.visible_lists[i][j]
                beta = len(self.visible_sets[vertex_index] - self.visible_sets[i])
                component_index = self.component_indices[i]
                divisor = len(self.connected_components[component_index] - self.visible_sets[i])
                if divisor > 0:
                    beta /= divisor
                else:
                    beta = np.inf
                self.beta_vws[i].append(beta)

        # print(self.beta_vws)

        # self.beta_lookout_size_table = [np.hstack((0.0, np.sort(beta_vw), 1.0))
                                        # for beta_vw in self.beta_vws]
        self.beta_lookout_size_table = [list(np.sort(beta_vw)) for beta_vw in self.beta_vws]
        for i in tqdm(range(len(self.beta_lookout_size_table))):
            self.beta_lookout_size_table[i].append(self.beta_lookout_size_table[i][-1])

        # print(self.beta_lookout_size_table)

        self.alpha_beta_pairs = [[] for _ in range(self.N)]
        for i in tqdm(range(self.N)):
            alphas = list(reversed(np.linspace(0, 1, len(self.beta_lookout_size_table[i]))))
            pairs = [(alpha, beta) for alpha, beta in zip(alphas, self.beta_lookout_size_table[i])]
            self.alpha_beta_pairs[i] = pairs

        # print(self.alpha_beta_pairs)

    def query_expansiveness(self, alpha, beta):
        for i in range(self.N):
            if not pareto_dominated(alpha, beta, self.alpha_beta_pairs[i]):
                # print(self.alpha_beta_pairs[i])
                return False
        return True

    def plot_pareto_scatter(self, ax, color="black"):
        pairs = np.array(self.get_all_alpha_beta_points())
        ax.scatter(pairs[:,0], pairs[:,1], c=color)

    def get_all_alpha_beta_points(self):
        pairs = []
        for i in range(len(self.alpha_beta_pairs)):
            for j in range(len(self.alpha_beta_pairs[i])):
                alpha1, beta1 = self.alpha_beta_pairs[i][j]
                pairs.append((alpha1, beta1))
        return pairs

    def plot_combined_pareto_curve(self, ax, color="black"):
        pareto_points = pareto_front(self.get_all_alpha_beta_points())
        self.plot_pareto_curve(pareto_points, ax, color)

    def plot_pareto_curve(self, pairs, ax, color="black"):
        alphas = []
        betas = []
        for j in range(len(pairs)):
            alpha1, beta1 = pairs[j]
            if j > 0:
                alpha0, beta0 = pairs[j-1]
                alphas.append(min(alpha0, alpha1))
                betas.append(min(beta0, beta1))
            alphas.append(alpha1)
            betas.append(beta1)
        ax.plot(alphas, betas, c=color)

    def plot_pareto_curves(self, ax, color="black"):
        for pairs in self.alpha_beta_pairs:
            self.plot_pareto_curve(pairs, ax, color)

def pareto_dominated(alpha_in, beta_in, frontier):
    for alpha, beta in frontier:
        if alpha_in <= alpha and beta_in <= beta:
            return True
    return False

# Thanks ChatGPT!
def pareto_front(points):
    # Sort points by x-coordinate (ascending), breaking ties by y-coordinate
    points = sorted(points, key=lambda p: (p[0], p[1]))

    pareto_points = []
    current_min_y = float('inf')  # Track the smallest y encountered

    # Traverse sorted points
    print("Computing pareto front...")
    for x, y in tqdm(points):
        if y < current_min_y:  # This point is not dominated
            pareto_points.append((x, y))
            current_min_y = y  # Update the smallest y

    return pareto_points

if __name__ == "__main__":
    # Fully connected graph. Should be (1, 1)-expansive.
    G = nx.complete_graph(3)
    expansiveness = Expansiveness(G)
    print(expansiveness.query_expansiveness(1, 1))

    # Line graph 0 -- 1 -- 2 -- 3.
    G = nx.path_graph(4)
    expansiveness = Expansiveness(G)
    print(not expansiveness.query_expansiveness(1, 1))
    print(expansiveness.query_expansiveness(1/3, 1/2))
    expansiveness.plot_pareto_curves()

    # Disconnected graph 0 -- 1    2 -- 3.
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(0, 1)
    G.add_edge(2, 3)
    expansiveness = Expansiveness(G)
    print(expansiveness.query_expansiveness(1, 1))