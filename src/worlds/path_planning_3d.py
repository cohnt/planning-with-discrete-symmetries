import numpy as np
import os.path
import time
import alphashape
import typing

from dataclasses import dataclass

from src.util import repo_dir
import src.symmetry as symmetry

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
    symmetry: symmetry.SymmetryGroupSO3Base
    dualshape: bool
    limits: np.ndarray[(3,2)]
    n_obstacle_points: int
    alpha: float
    seed: int
# dualshape is only needed for the platonic solids.
# If True, return Octahedron and Icosahedron.
# If False, return Cube and Dodecahedron.

def compute_face_normal(a, b, c):
    return np.cross(b - a, c - a)

def make_tetrahedron_obj_sdf(vertices, name, path):
    assert len(vertices) == 4
    for v in vertices:
        assert len(v) == 3

    # One possible ordering of the vertices for each face.
    faces = [
        np.array([1, 3, 2]),
        np.array([1, 4, 3]),
        np.array([1, 2, 4]),
        np.array([2, 3, 4])
    ]

    # Compute face normals, to check if we have to flip the ordering.
    face_normals = []
    for face in faces:
        face_normals.append(compute_face_normal(*(vertices[face-1,:])))

    centroid = np.mean(vertices, axis=0)
    flip = []
    for face_normal, face in zip(face_normals, faces):
        flip.append(
            np.dot(face_normal - centroid, vertices[face[0]] - centroid) < 0
        )

    for i in range(len(flip)):
        if flip[i]:
            faces[i] = tuple(reversed(faces[i]))
        else:
            faces[i] = tuple(faces[i])

    with open(os.path.join(path, name) + ".obj", "w") as f:
        f.write("g %s\n" % name)
        f.write("\n")
        for v in vertices:
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
                <ambient>1 0.5 0.5 0.5</ambient>
                <diffuse>1 0.5 0.5 0.5</diffuse>
                <specular>1 0.5 0.5 0.5</specular>
                <emissive>1 0.5 0.5 0.5</emissive>
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
    if add_limits_as_obstacles:
        raise NotImplementedError

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = world_limits

    np.random.seed(seed)
    points = np.random.uniform(low=[xmin, ymin, zmin],
                               high=[xmax, ymax, zmax],
                               size=(n_points, 3))
    gen = alphashape.alphasimplices(points)

    tets = []
    for simplex, r, in gen:
        if r < 1/alpha:
            tets.append(points[simplex])

    return tets

def add_obstacles_to_directives(directives, tets, path):
    full_path = os.path.join(repo_dir(), path)
    directives_str = directives
    for idx, tet in enumerate(tets):
        name = "tet_%d" % idx
        make_tetrahedron_obj_sdf(tet, name, full_path)
        directives_str += """
- add_model:
    name: %s
    file: package://symmetries/%s.sdf

- add_weld:
    parent: world
    child: %s::base
""" % (name, os.path.join(path, name), name)
    return directives_str

def params_to_shape(params : SetupParams):
    shape = None

    if isinstance(params.symmetry, symmetry.CyclicGroupSO3):
        # n-Gon Base Pyramid
        n_sides = param.symmetry.order()
        if n_sides == 1:
            raise NotImplementedError # TODO
        elif n_sides == 2:
            shape = "pyramids/rectangular_pyramid.sdf"
        elif n_sides == 3:
            raise NotImplementedError # Not planned -- use a tetrahedron
        elif n_sides == 4:
            raise NotImplementedError # TODO
        elif n_sides == 5:
            shape = "pyramids/pentagonal_pyramid.sdf"
        elif n_sides == 6:
            shape = "pyramids/hexagonal_pyramid.sdf"
        elif n_sides == 7:
            shape = "pyramids/heptagonal_pyramid.sdf"
        elif n_sides == 8:
            shape = "pyramids/octagonal_pyramid.sdf"
        else:
            raise NotImplementedError # Not planned
    elif isinstance(params.symmetry, symmetry.DihedralGroup):
        # n-Gon Prism
        n_sides = param.symmetry.order() // 2
        if n_sides == 1:
            raise NotImplementedError # TODO
        elif n_sides == 2:
            shape = "prisms/rectangular_prism.sdf"
        elif n_sides == 3:
            shape = "prisms/triangular_prism.sdf"
        elif n_sides == 4:
            raise NotImplementedError # Not planned -- use a cube
        elif n_sides == 5:
            shape = "prisms/pentagonal_prism.sdf"
        elif n_sides == 6:
            shape = "prisms/hexagonal_prism.sdf"
        elif n_sides == 7:
            shape = "prisms/heptagonal_prism.sdf"
        elif n_sides == 8:
            shape = "prisms/octagonal_prism.sdf"
        else:
            raise NotImplementedError # Not planned
    elif isinstance(params.symmetry, symmetry.TetrahedralGroup):
        # Tetrahedron
        shape = "tetrahedron.sdf"
    elif isinstance(params.symmetry, symmetry.OctahedralGroup):
        if params.dualshape:
            shape = "octahedron.sdf"
        else:
            shape = "cube.sdf"
    elif isinstance(params.symmetry, symmetry.IcosahedralGroup):
        if params.dualshape:
            shape = "icosahedron.sdf"
        else:
            shape = "dodecahedron.sdf"
    else:
        raise NotImplementedError

    return shape

def build_env(meshcat, params : SetupParams):
    shape = params_to_shape(params)

    directives_str = f"""directives:
- add_model:
    name: robot
    file: package://symmetries/models/{shape}
- add_weld:
    parent: world
    child: robot::base
"""
    
    tets = alphashape_make_obstacles(params.limits,
                                     n_points=params.n_obstacle_points,
                                     alpha=params.alpha,
                                     seed=params.seed,
                                     add_limits_as_obstacles=False)

    builder = RobotDiagramBuilder(0)
    builder.parser().package_map().Add("symmetries", repo_dir())

    directives_str = add_obstacles_to_directives(directives_str, tets, "models/dynamically_generated")
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

    meshcat.SetProperty("/Grid", "visible", False)

    camera_target = np.mean(params.limits, axis=1)
    camera_position = camera_target * 2.5
    meshcat.SetCameraPose(camera_position, camera_target)

    return diagram, collision_checker

if __name__ == "__main__":
    meshcat = StartMeshcat()

    limits = [[0, 20], [0, 20], [0, 20]]
    params = SetupParams(symmetry.TetrahedralGroup(), True, limits, 300, 0.45, 0)
    diagram, collision_checker = build_env(meshcat, params)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

    diagram.ForcedPublish(diagram_context)

    while True:
        time.sleep(1)