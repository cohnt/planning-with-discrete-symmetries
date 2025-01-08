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
    limits: np.ndarray[(2,2,2)]
    n_obstacle_points: int
    alpha: float
    seed: int

def make_tetrahedron_obj_sdf(vertices, name, path):
    assert len(vertices) == 4
    for v in vertices:
        assert len(v) == 3
    
    pass

def alphashape_make_obstacles(world_limits, n_points=40, alpha=1.5, seed=2, add_limits_as_obstacles=False):
    pass

def add_obstacles_to_directives(directives, tris, path):
    pass

def build_env(meshcat, params : SetupParams):
    pass

if __name__ == "__main__":
    pass