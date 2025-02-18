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

def draw_path(meshcat, path_vertices, indices, path="rrt", linewidth=1.0, color=Rgba(0, 0, 0, 1)):
    graph = nx.DiGraph()
    for i, v in enumerate(path_vertices):
        graph.add_node(i, q=v)
    for i in range(1, len(path_vertices)):
        graph.add_edge(i-1, i)

    draw_graph(meshcat, graph, indices, path, linewidth, color)