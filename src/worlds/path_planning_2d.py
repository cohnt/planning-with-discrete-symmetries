import numpy as np
import os.path
import time
import alphashape
import typing

from dataclasses import dataclass

from src.util import repo_dir

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
    RotationMatrix,
    AddDefaultVisualization,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker
)

@dataclass
class SetupParams:
    n_sides: int
    limits: np.ndarray[(2,2)]
    n_obstacle_points: int
    alpha: float
    seed: int

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
    faces.append((2, 6, 4)) # Reversed for outward-facing normal

    # Sides
    faces.append((1, 2, 3))
    faces.append((2, 4, 3)) # Reversed for outward-facing normal
    faces.append((3, 4, 5))
    faces.append((4, 6, 5)) # Reversed for outward-facing normal
    faces.append((5, 6, 1))
    faces.append((6, 2, 1)) # Reversed for outward-facing normal

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

def draw_boundaries(meshcat, limits):
    vertices = [
        [limits[0][0], limits[1][0]],
        [limits[0][0], limits[1][1]],
        [limits[0][1], limits[1][1]],
        [limits[0][1], limits[1][0]]
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0)
    ]

    # Prepare points for all line segments
    points = []

    # Add line segments to Meshcat
    for start_idx, end_idx in edges:
        start = vertices[start_idx]
        end = vertices[end_idx]
        points.append(start + [0])
        points.append(end + [0])

    # Convert points and colors to arrays
    points = np.vstack(points).T  # 3xN array of points
    starts = points[:,::2]
    ends = points[:,1::2]

    # Add line segments to the group "boundaries"
    meshcat.SetLineSegments(
        path="boundaries",
        start=starts,
        end=ends
    )

def build_env(meshcat, params : SetupParams):
    if params.n_sides == 1:
        raise NotImplementedError # TODO
    elif params.n_sides == 2:
        shape = "prisms/rectangular_prism_2d.sdf"
    elif params.n_sides == 3:
        shape = "prisms/triangular_prism_2d.sdf"
    elif params.n_sides == 4:
        shape = "prisms/square_prism_2d.sdf"
    elif params.n_sides == 5:
        shape = "prisms/pentagonal_prism_2d.sdf"
    elif params.n_sides == 6:
        shape = "prisms/hexagonal_prism_2d.sdf"
    elif params.n_sides == 7:
        shape = "prisms/heptagonal_prism_2d.sdf"
    elif params.n_sides == 8:
        shape = "prisms/octagonal_prism_2d.sdf"
    else:
        raise NotImplementedError # Not planned

    directives_str = f"""directives:
- add_model:
    name: robot
    file: package://symmetries/models/{shape}
- add_weld:
    parent: world
    child: robot::base
"""

    tris = alphashape_make_obstacles(params.limits,
                                     n_points=params.n_obstacle_points,
                                     alpha=params.alpha,
                                     seed=params.seed,
                                     add_limits_as_obstacles=False)

    builder = RobotDiagramBuilder(0)
    builder.parser().package_map().Add("symmetries", repo_dir())

    directives_str = add_obstacles_to_directives(directives_str, tris, "models/dynamically_generated")
    directives = LoadModelDirectivesFromString(directives_str)
    ProcessModelDirectives(directives, builder.plant(), builder.parser())

    # AddDefaultVisualization(builder.builder(), meshcat)
    meshcat_visual_params = MeshcatVisualizerParams()
    meshcat_visual_params.delete_on_initialization_event = False
    meshcat_visual_params.role = Role.kIllustration
    meshcat_visual_params.prefix = "visual"
    meshcat_visual_params.visible_by_default = True
    meshcat_visual = MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_visual_params)
    meshcat_collision_params = MeshcatVisualizerParams()
    meshcat_collision_params.delete_on_initialization_event = False
    meshcat_collision_params.role = Role.kProximity
    meshcat_collision_params.prefix = "collision"
    meshcat_collision_params.visible_by_default = False
    meshcat_collision = MeshcatVisualizer.AddToBuilder(
        builder.builder(), builder.scene_graph(), meshcat, meshcat_collision_params)

    diagram = builder.Build()

    model = diagram
    robot_model_instances = [diagram.plant().GetModelInstanceByName("robot")]
    edge_step_size = 0.01
    collision_checker = SceneGraphCollisionChecker(model=model, robot_model_instances=robot_model_instances, edge_step_size=edge_step_size)

    rot = np.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0]
    ])
    # print(np.linalg.det(rot))
    trans = np.array([0, 0, 1])
    tf = RigidTransform(RotationMatrix(rot), trans)

    meshcat.Set2dRenderMode(X_WC=tf,
                            xmin=params.limits[0][0] - 1,
                            xmax=params.limits[0][1] + 1,
                            ymin=params.limits[1][0] - 1,
                            ymax=params.limits[1][1] + 1)
    meshcat.SetProperty("/Lights/PointLightNegativeX", "visible", False)
    meshcat.SetProperty("/Lights/PointLightPositiveX", "visible", False)
    meshcat.SetProperty("/Lights/FillLight", "visible", False)

    draw_boundaries(meshcat, params.limits)

    return diagram, collision_checker

if __name__ == "__main__":
    meshcat = StartMeshcat()

    limits = [[0, 20], [0, 20]]
    params = SetupParams(3, limits, 200, 1, 0)
    diagram, collision_checker = build_env(meshcat, params)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

    diagram.plant().SetPositions(plant_context, [10, 10, 0])
    diagram.ForcedPublish(diagram_context)
    time.sleep(5)

    while True:
        for i in range(100):
            diagram.plant().SetPositions(plant_context, [i / 100. * 20, i / 100. * 10, i / 100. * 2 * np.pi])
            diagram.ForcedPublish(diagram_context)
            time.sleep(0.1)