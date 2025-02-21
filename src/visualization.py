import numpy as np
import networkx as nx

from pydrake.all import Rgba

def draw_graph(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1)):
    assert len(indices) in [2, 3]

    if len(indices) == 2:
        draw_graph_2d(meshcat, graph, indices, path, linewidth, color)
    else:
        draw_graph_3d(meshcat, graph, indices, path, linewidth, color)

def draw_graph_2d(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1)):
    nodes = [graph.nodes[i]["q"][indices] for i in range(len(graph.nodes))]
    N = len(nodes)

    vertices = []
    faces = []

    for i in range(0, N-1):
        edge = nodes[i+1] - nodes[i]
        orth = np.array([edge[1], -edge[0]])
        orth /= np.linalg.norm(orth)
        orth *= linewidth
        orth /= 2
        vertices.append(nodes[i])
        vertices.append(nodes[i] + orth)
        vertices.append(nodes[i] - orth)
        vertices.append(nodes[i+1])
        vertices.append(nodes[i+1] + orth)
        vertices.append(nodes[i+1] - orth)

    foo = np.zeros((len(vertices), 3))
    foo[:,:2] = np.array(vertices)
    vertices = foo

    faces = []
    for i in range(N-1):
        idx = 6 * i
        faces.append([idx, idx+1, idx+3])
        faces.append([idx, idx+2, idx+3])
        faces.append([idx+1, idx+3, idx+4])
        faces.append([idx+2, idx+3, idx+5])

        if i > 0:
            faces.append([idx-2, idx-1, idx+1])
            faces.append([idx-1, idx+1, idx+2])

    faces = np.array(faces)

    meshcat.SetTriangleMesh(
        path=path,
        vertices=vertices.T,
        faces=faces.T,
        rgba=color)

def draw_graph_3d(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1)):
    nodes = [graph.nodes[i]["q"][indices] for i in range(len(graph.nodes))]
    N = len(nodes)

    vertices = []
    faces = []

    orth1 = orth2 = None
    for i in range(0, N-1):
        edge = nodes[i+1] - nodes[i]
        edge_unit = edge / np.linalg.norm(edge)

        if orth1 is None:
            orth1 = np.random.randn(3)
        orth1 -= orth1.dot(edge_unit) * edge_unit
        orth1 /= np.linalg.norm(orth1)
        orth1 *= linewidth
        orth1 /= 2

        orth2 = np.cross(edge, orth1)
        orth2 /= np.linalg.norm(orth2)
        orth2 *= linewidth
        orth2 /= 2

        vertices.append(nodes[i] + orth1)
        vertices.append(nodes[i] - orth1)
        vertices.append(nodes[i] + orth2)
        vertices.append(nodes[i] - orth2)

        vertices.append(nodes[i+1] + orth1)
        vertices.append(nodes[i+1] - orth1)
        vertices.append(nodes[i+1] + orth2)
        vertices.append(nodes[i+1] - orth2)

    vertices = np.array(vertices)

    faces = []
    for i in range(N-1):
        idx = 8 * i

        # +1, +2
        faces.append([idx, idx+2, idx+4])
        faces.append([idx+2, idx+4, idx+6])

        # +1, -2
        faces.append([idx, idx+3, idx+4])
        faces.append([idx+3, idx+4, idx+7])

        # -1, +2
        faces.append([idx+1, idx+2, idx+5])
        faces.append([idx+2, idx+5, idx+6])

        # -1, -2
        faces.append([idx+1, idx+3, idx+5])
        faces.append([idx+3, idx+5, idx+7])

        if idx > 0:
            idx -= 4

            # +1, +2
            faces.append([idx, idx+2, idx+4])
            faces.append([idx+2, idx+4, idx+6])

            # +1, -2
            faces.append([idx, idx+3, idx+4])
            faces.append([idx+3, idx+4, idx+7])

            # -1, +2
            faces.append([idx+1, idx+2, idx+5])
            faces.append([idx+2, idx+5, idx+6])

            # -1, -2
            faces.append([idx+1, idx+3, idx+5])
            faces.append([idx+3, idx+5, idx+7])

    faces.append([0, 1, 2])
    faces.append([0, 1, 3])

    faces.append(np.array([-1, -2, -3]) + len(vertices))
    faces.append(np.array([-1, -2, -4]) + len(vertices))

    faces = np.array(faces)

    meshcat.SetTriangleMesh(
        path=path,
        vertices=vertices.T,
        faces=faces.T,
        rgba=color)

def draw_path(meshcat, path_vertices, indices, path="rrt", linewidth=1.0, color=Rgba(0, 0, 0, 1)):
    graph = nx.DiGraph()
    for i, v in enumerate(path_vertices):
        graph.add_node(i, q=v)
    for i in range(1, len(path_vertices)):
        graph.add_edge(i-1, i)

    draw_graph(meshcat, graph, indices, path, linewidth, color)