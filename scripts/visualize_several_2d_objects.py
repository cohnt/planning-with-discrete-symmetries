import sys
sys.path.append("..")

import time

import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import StartMeshcat

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]

for n_sides in [2, 3, 4, 5, 6, 7, 8]:
    params = path_planning_2d.SetupParams(n_sides, limits, 200, 1, 0)
    diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

    q = [10, 10, 0] # Put the object in collision-free space

    diagram.plant().SetPositions(plant_context, q)
    diagram.ForcedPublish(diagram_context)
    time.sleep(0.01)