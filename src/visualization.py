import numpy as np
import networkx as nx

from pydrake.all import Rgba, Ellipsoid

def draw_graph(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1), draw_caps=True):
    assert len(indices) in [2, 3]

    if len(indices) == 2:
        draw_graph_2d(meshcat, graph, indices, path, linewidth, color, draw_caps)
    else:
        draw_graph_3d(meshcat, graph, indices, path, linewidth, color, draw_caps)

def draw_graph_2d(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1), draw_caps=True):
    nodes = [graph.nodes[i]["q"][indices] for i in range(len(graph.nodes))]
    N = len(nodes)

    vertices = []
    faces = []
    edge_points = set()

    # Folder to hold everything
    meshcat.SetTransform(path, np.eye(4))  # just ensures the path exists

    for u, v in graph.edges:
        p, q = nodes[u], nodes[v]
        edge_points.add(tuple(p))
        edge_points.add(tuple(q))

        edge = q - p
        orth = np.array([edge[1], -edge[0]])
        orth /= np.linalg.norm(orth)
        orth *= linewidth / 2

        base_idx = len(vertices)
        vertices.extend([
            [p[0], p[1], 0],
            [p[0]+orth[0], p[1]+orth[1], 0],
            [p[0]-orth[0], p[1]-orth[1], 0],
            [q[0], q[1], 0],
            [q[0]+orth[0], q[1]+orth[1], 0],
            [q[0]-orth[0], q[1]-orth[1], 0],
        ])

        faces.extend([
            [base_idx, base_idx+1, base_idx+3],
            [base_idx, base_idx+2, base_idx+3],
            [base_idx+1, base_idx+3, base_idx+4],
            [base_idx+2, base_idx+3, base_idx+5],
        ])

    vertices = np.array(vertices)
    faces = np.array(faces)

    meshcat.SetTriangleMesh(
        path=path + "/edges",
        vertices=vertices.T,
        faces=faces.T,
        rgba=color)

    if draw_caps:
        sphere_radius = linewidth / 2
        for i, point in enumerate(edge_points):
            x, y = point
            ellipsoid = Ellipsoid(sphere_radius, sphere_radius, sphere_radius)
            meshcat.SetObject(f"{path}/caps_{i}", ellipsoid, color)
            # Move it to the vertex
            meshcat.SetTransform(f"{path}/caps_{i}", np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]))

def draw_graph_3d(meshcat, graph, indices, path="rrt", linewidth=0.1, color=Rgba(0, 0, 0, 1), draw_caps=True):
    nodes = [graph.nodes[i]["q"][indices] for i in range(len(graph.nodes))]

    vertices = []
    faces = []

    # Folder for the whole graph
    meshcat.SetTransform(path, np.eye(4))

    # Keep track of vertices incident to edges
    edge_points = set()

    # Iterate over all edges in the graph
    for u, v in graph.edges:
        p, q = nodes[u], nodes[v]
        edge_points.add(tuple(p))
        edge_points.add(tuple(q))

        edge = q - p
        edge_unit = edge / np.linalg.norm(edge)

        # Generate two orthogonal vectors perpendicular to edge
        orth1 = np.random.randn(3)
        orth1 -= orth1.dot(edge_unit) * edge_unit
        orth1 /= np.linalg.norm(orth1)
        orth1 *= linewidth / 2

        orth2 = np.cross(edge_unit, orth1)
        orth2 /= np.linalg.norm(orth2)
        orth2 *= linewidth / 2

        base_idx = len(vertices)
        vertices.extend([
            p + orth1, p - orth1, p + orth2, p - orth2,
            q + orth1, q - orth1, q + orth2, q - orth2
        ])

        faces.extend([
            # +1, +2
            [base_idx, base_idx+2, base_idx+4],
            [base_idx+2, base_idx+4, base_idx+6],
            # +1, -2
            [base_idx, base_idx+3, base_idx+4],
            [base_idx+3, base_idx+4, base_idx+7],
            # -1, +2
            [base_idx+1, base_idx+2, base_idx+5],
            [base_idx+2, base_idx+5, base_idx+6],
            # -1, -2
            [base_idx+1, base_idx+3, base_idx+5],
            [base_idx+3, base_idx+5, base_idx+7],
        ])

    # Draw triangle mesh for edges
    vertices = np.array(vertices)
    faces = np.array(faces)
    meshcat.SetTriangleMesh(
        path + "/edges",
        vertices.T,
        faces.T,
        rgba=color
    )

    # Draw spherical caps at vertices
    if draw_caps:
        sphere_radius = linewidth / 2
        for i, point in enumerate(edge_points):
            x, y, z = point
            ellipsoid = Ellipsoid(sphere_radius, sphere_radius, sphere_radius)
            meshcat.SetObject(f"{path}/caps_{i}", ellipsoid, color)
            meshcat.SetTransform(f"{path}/caps_{i}", np.array([
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1]
            ]))

def draw_path(meshcat, path_vertices, indices, path="rrt", linewidth=1.0, color=Rgba(0, 0, 0, 1)):
    graph = nx.DiGraph()
    for i, v in enumerate(path_vertices):
        graph.add_node(i, q=v)
    for i in range(1, len(path_vertices)):
        graph.add_edge(i-1, i)

    draw_graph(meshcat, graph, indices, path, linewidth, color)