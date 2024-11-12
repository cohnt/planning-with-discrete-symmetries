import sys
sys.path.append("..")

import time
import numpy as np

import src.planners.imacs as imacs
import src.planners.prm as prm
import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import (
    StartMeshcat
)

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(3, limits, 200, 1, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

G = symmetry.CyclicGroupSO2(3)
Sampler = imacs.SO2SampleUniform(G, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric = imacs.SO2DistanceSq(G, 3, 2)
Interpolator = imacs.SO2Interpolate(G, 3, 2)
options = prm.PRMOptions()
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionChecker, options)

roadmap.build()

q = roadmap.nodes[0]

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

while True:
    for q in roadmap.nodes:
        diagram.plant().SetPositions(plant_context, q)
        diagram.ForcedPublish(diagram_context)
        time.sleep(1)