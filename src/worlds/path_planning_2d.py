import numpy as np
import os.path
import time
import pathlib
import alphashape

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    LoadModelDirectivesFromString,
    ProcessModelDirectives,
    MeshcatVisualizerParams,
    Role,
    MeshcatVisualizer,
    RotationMatrix,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix
)

def make_triangle_obj_sdf(vertices, name, path):
    assert len(vertices) == 3
    for v in vertices:
        assert len(v) == 2
    full_vertices = []
    for v in vertices:
        full_vertices.append([v[0], v[1], 0.5])
        full_vertices.append([v[0], v[1], -0.5])

    # obj files use 1-indexing
    faces = []

    # Top and bottom
    faces.append((1, 3, 5))
    faces.append((2, 4, 6))

    # Sides
    faces.append((1, 2, 3))
    faces.append((2, 3, 4))
    faces.append((3, 4, 5))
    faces.append((4, 5, 6))
    faces.append((5, 6, 1))
    faces.append((6, 1, 2))

    with open(os.path.join(path, name) + ".obj", "w") as f:
        f.write("g %s\n" % name)
        f.write("\n")
        for v in full_vertices:
            f.write("v %f %f %f\n" % tuple(v))
        f.write("\n")
        for face in faces:
            f.write("f %d %d %d\n" % face)

    with open(os.path.join(path, name) + ".sdf", "w") as f:
        f.write("""<?xml version="1.0"?>
<sdf version="1.7">
<model name="%s">
    <link name="base">
        <pose>0 0 0 0 0 0</pose>
        <visual name="visual">
            <geometry>
                <mesh>
                    <scale>1 1 1</scale>
                    <uri>%s.obj</uri>
                </mesh>
            </geometry>
            <material>
                <ambient>1 0.5 0.5 1</ambient>
                <diffuse>1 0.5 0.5 1</diffuse>
                <specular>1 0.5 0.5 1</specular>
                <emissive>1 0.5 0.5 1</emissive>
            </material>
        </visual>
        <collision name="base_collision">
            <geometry>
                <mesh>
                    <scale>1 1 1</scale>
                    <uri>%s.obj</uri>
                </mesh>
            </geometry>
        </collision>
    </link>
</model>
</sdf>""" % (name, name, name))

def alphashape_make_obstacles(world_limits, n_points=40, alpha=1.5, seed=2, add_limits_as_obstacles=False):
    (xmin, xmax), (ymin, ymax) = world_limits
    sw = np.array([xmin, ymin])
    se = np.array([xmax, ymin])
    nw = np.array([xmin, ymax])
    ne = np.array([xmax, ymax])
    s = np.array([(xmin+xmax)*0.5, ymin-0.5])
    n = np.array([(xmin+xmax)*0.5, ymax+0.5])
    w = np.array([xmin-0.5, (ymin+ymax)*0.5])
    e = np.array([xmax+0.5, (ymin+ymax)*0.5])
    
    np.random.seed(seed)

    points = np.random.uniform(low=sw, high=ne, size=(n_points, 2))
    gen = alphashape.alphasimplices(points)

    tris = []
    for simplex, r in gen:
        if r < 1/alpha:
            tris.append(points[simplex])

    if add_limits_as_obstacles:
        tris.append(np.array([sw, se, s]))
        tris.append(np.array([se, ne, e]))
        tris.append(np.array([ne, nw, n]))
        tris.append(np.array([nw, sw, w]))

    return tris

def add_obstacles_to_directives(directives, tris, path):
    full_path = os.path.join(repo_dir(), path)
    directives_str = directives
    for idx, tri in enumerate(tris):
        name = "tri_%d" % idx
        make_triangle_obj_sdf(tri, name, full_path)
        directives_str += """
- add_model:
    name: %s
    file: package://symmetries/%s.sdf

- add_weld:
    parent: world
    child: %s::base
""" % (name, os.path.join(path, name), name)
    return directives_str

def to_pose_3d(pose_2d):
    # Given (x, y, theta), return (x, y, z, r, p, y), with z=r=p=0, and y=theta
    # (Note that the prism models assume the x axis is the axis of symmetry)
    return np.array([pose_2d[0], pose_2d[1], 0, 0, 0, pose_2d[2]])

if __name__ == "__main__":
    directives_str = """directives:
- add_model:
    name: triangle
    file: package://symmetries/models/prisms/triangular_prism_2d.sdf
- add_weld:
    parent: world
    child: triangle::base
"""

    limits = np.array([[0, 20], [0, 20]])
    tris = alphashape_make_obstacles(limits, n_points=200, alpha=1., seed=0, add_limits_as_obstacles=False)

    from src.util import repo_dir
    path = "models/dynamically_generated"
    directives_str = add_obstacles_to_directives(directives_str, tris, path)

    # print(directives_str)
    
    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    parser = Parser(plant)
    parser.package_map().Add("symmetries", repo_dir())
    directives = LoadModelDirectivesFromString(directives_str)
    models = ProcessModelDirectives(directives, plant, parser)

    plant.Finalize()

    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    meshcat_visual_params.visible_by_default = True
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visual_params)

    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision"
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_collision_params)

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    rot = np.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0]
    ])
    # print(np.linalg.det(rot))
    trans = np.array([0, 0, 1])
    tf = RigidTransform(RotationMatrix(rot), trans)

    meshcat.Set2dRenderMode(X_WC=tf, xmin=limits[0][0], xmax=limits[0][1], ymin=limits[1][0], ymax=limits[1][1])
    meshcat.SetProperty("/Lights/PointLightNegativeX", "visible", False)
    meshcat.SetProperty("/Lights/PointLightPositiveX", "visible", False)
    meshcat.SetProperty("/Lights/FillLight", "visible", False)

    plant.SetPositions(plant_context, to_pose_3d([10, 10, 0]))
    diagram.ForcedPublish(diagram_context)
    time.sleep(5)

    while True:
        for i in range(100):
            plant.SetPositions(plant_context, to_pose_3d([i / 100. * 20, i / 100. * 10, i / 100. * 2 * np.pi]))
            diagram.ForcedPublish(diagram_context)
            time.sleep(0.1)