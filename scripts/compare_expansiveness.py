import sys
sys.path.append("..")

import numpy as np

import src.planners.imacs as imacs
import src.planners.prm as prm
import src.symmetry as symmetry
import src.expansiveness as expansiveness
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import StartMeshcat

options = prm.PRMOptions(max_vertices=200, neighbor_radius=np.inf, neighbor_mode="radius")

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(3, limits, 200, 1.25, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

G1 = symmetry.CyclicGroupSO2(3)
Sampler1 = imacs.SO2SampleUniform(G1, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric1 = imacs.SO2DistanceSq(G1, 3, 2)
Interpolator1 = imacs.SO2Interpolate(G1, 3, 2)
roadmap1 = prm.PRM(Sampler1, Metric1, Interpolator1, CollisionChecker, options)

np.random.seed(0)
roadmap1.build()

G2 = symmetry.CyclicGroupSO2(1)
Sampler2 = imacs.SO2SampleUniform(G2, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
Metric2 = imacs.SO2DistanceSq(G2, 3, 2)
Interpolator2 = imacs.SO2Interpolate(G2, 3, 2)
roadmap2 = prm.PRM(Sampler2, Metric2, Interpolator2, CollisionChecker, options)

np.random.seed(0)
roadmap2.build()

e1 = expansiveness.Expansiveness(roadmap1.graph)
e2 = expansiveness.Expansiveness(roadmap2.graph)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))

e1.plot_pareto_curves(ax, "blue")
e2.plot_pareto_curves(ax, "red")

plt.show()

import pdb
pdb.set_trace()