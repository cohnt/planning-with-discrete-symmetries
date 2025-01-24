import numpy as np
import networkx as nx

from pydrake.all import Rgba

def draw_graph(meshcat, graph, indices, path="rrt", linewidth=1.0, color=Rgba(0, 0, 0, 1)):
    assert len(indices) in [2, 3]
    nodes = [graph.nodes[i]["q"][indices] for i in range(len(graph.nodes))]

    if len(indices) == 2:
        nodes = [np.append(q, 0) for q in nodes]

    starts = np.array([nodes[i] for i, _ in graph.edges]).T
    ends   = np.array([nodes[j] for _, j in graph.edges]).T

    meshcat.SetLineSegments(
        path=path,
        start=starts,
        end=ends,
        line_width=linewidth,
        rgba=color)