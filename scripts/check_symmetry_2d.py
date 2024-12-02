import sys
sys.path.append("..")

import numpy as np
import pathlib
import time
from scipy.stats import special_ortho_group

import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d

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
)

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(4, limits, 200, 1, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)
plant = diagram.plant()

G = symmetry.CyclicGroupSO2(4)

button = meshcat.AddButton("Next")
num_clicks = 0
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)

mat = np.eye(2)
mat = special_ortho_group.rvs(2)
tfs_so2 = G.orbit(mat)

thetas = np.linspace(0, 2 * np.pi, G.order(), endpoint=False)
for theta in thetas:
    positions = np.array([1, 1, theta])
    plant.SetPositions(plant_context, positions)

    diagram.ForcedPublish(diagram_context)

    while meshcat.GetButtonClicks("Next") == num_clicks:
        time.sleep(0.05)

    num_clicks += 1

meshcat.DeleteButton("Next")

print("Done! The object should not have appeared to move at all.")