import sys
sys.path.append("..")

import numpy as np

import src.planners.imacs as imacs
import src.planners.prm as prm
import src.symmetry as symmetry
import src.expansiveness as expansiveness
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import StartMeshcat

options = prm.PRMOptions(max_vertices=500, neighbor_radius=np.inf, neighbor_mode="radius")

meshcat = StartMeshcat()
animate = False

n = 3
limits = [[0, 5], [0, 5]]
params = path_planning_2d.SetupParams(n, limits, 8, 1.05, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)

for seed in range(10):
    G1 = symmetry.CyclicGroupSO2(n)
    Sampler1 = imacs.SO2SampleUniform(G1, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
    Metric1 = imacs.SO2DistanceSq(G1, 3, 2)
    Interpolator1 = imacs.SO2Interpolate(G1, 3, 2)
    roadmap1 = prm.PRM(Sampler1, Metric1, Interpolator1, CollisionChecker, options)

    np.random.seed(seed)
    roadmap1.build()

    G2 = symmetry.CyclicGroupSO2(1)
    Sampler2 = imacs.SO2SampleUniform(G2, 3, 2, [limits[0][0], limits[1][0], 0], [limits[0][1], limits[1][1], 0])
    Metric2 = imacs.SO2DistanceSq(G2, 3, 2)
    Interpolator2 = imacs.SO2Interpolate(G2, 3, 2)
    roadmap2 = prm.PRM(Sampler2, Metric2, Interpolator2, CollisionChecker, options)

    np.random.seed(seed)
    roadmap2.build()

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel("alpha")
    plt.ylabel("beta")

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    e1 = expansiveness.Expansiveness(roadmap1.graph)
    e2 = expansiveness.Expansiveness(roadmap2.graph)

    alpha_query = 0.4
    beta_query = 0.02
    print()
    print("Symmetry aware (0.5, 0.02)-expansive?", e1.query_expansiveness(alpha_query, beta_query))
    print("Symmetry unaware (0.5, 0.02)-expansive?", e2.query_expansiveness(alpha_query, beta_query))
    print()

    e1.plot_combined_pareto_curve(ax, "blue")
    e2.plot_combined_pareto_curve(ax, "red")
    ax.scatter([alpha_query], [beta_query], color="black", marker="x")

    blue_patch = mpatches.Patch(color='blue', label='Symmetry-Aware Planner')
    red_patch = mpatches.Patch(color='red', label='Symmetry-Unaware Planner')
    black_patch = mpatches.Patch(color="black", label="Query Point")
    plt.legend(handles=[blue_patch, red_patch, black_patch])

    plt.savefig("data/expansiveness_seed_%d.svg" % seed)
    plt.close()

    # plt.show()

# vals = range(options.max_vertices) if animate else [options.max_vertices]
# for i in vals:
#     plt.cla()
#     plt.xlabel("alpha")
#     plt.ylabel("beta")

#     i = int(i)

#     e1 = expansiveness.Expansiveness(roadmap1.graph.subgraph(range(0, i)))
#     e2 = expansiveness.Expansiveness(roadmap2.graph.subgraph(range(0, i)))

#     ax.set_xlim((0, 1))
#     ax.set_ylim((0, 1))

#     # e1.plot_pareto_scatter(ax, "blue")
#     # e2.plot_pareto_scatter(ax, "red")

#     # e1.plot_pareto_curves(ax, "blue")
#     # e2.plot_pareto_curves(ax, "red")

#     e1.plot_combined_pareto_curve(ax, "blue")
#     e2.plot_combined_pareto_curve(ax, "red")

#     blue_patch = mpatches.Patch(color='blue', label='Symmetry-Aware Planner')
#     red_patch = mpatches.Patch(color='red', label='Symmetry-Unaware Planner')
#     plt.legend(handles=[blue_patch, red_patch])

#     plt.draw()
#     plt.pause(0.01)

#     if not animate:
#         plt.show()
#         break

# if animate:
#     plt.show()