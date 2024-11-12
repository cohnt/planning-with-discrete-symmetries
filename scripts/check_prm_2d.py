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
options = prm.PRMOptions(max_vertices=1e2)
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionChecker, options)

roadmap.build()

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)

i = 0
q = roadmap.graph.nodes[i]["q"]
diagram.plant().SetPositions(plant_context, q)
diagram.ForcedPublish(diagram_context)

while True:
    i_next = np.random.choice(list(roadmap.graph.neighbors(i)))
    q_next = roadmap.graph[i][i_next]["qj"]
    
    qs = [roadmap.Interpolator(q, q_next, t) for t in np.linspace(0, 1, 20)]
    # print()
    print(i, i_next)
    # print(q, q_next)
    for qi in qs:
        # print(qi)
        diagram.plant().SetPositions(plant_context, qi)
        diagram.ForcedPublish(diagram_context)
        time.sleep(0.1)
    # print("dtheta", (q_next[2] - q[2]))

    i = i_next
    q = roadmap.graph.nodes[i]["q"]