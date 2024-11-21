import sys
sys.path.append("..")

import time

import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import StartMeshcat

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]

for seed in range(10):
    params = path_planning_2d.SetupParams(3, limits, 200, 1, seed)
    diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

    q = [-5, -5, 0] # Put the object off-screen

    diagram.plant().SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)
    time.sleep(1)