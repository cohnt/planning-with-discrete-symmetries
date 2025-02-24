import sys
sys.path.append("..")

import time
import numpy as np

import src.asymptotic_optimality_parameters as asymptotic
import src.planners.imacs as imacs
import src.planners.prm as prm
import src.symmetry as symmetry
import src.worlds.path_planning_2d as path_planning_2d

from pydrake.all import (
    StartMeshcat,
    PiecewisePolynomial,
    CompositeTrajectory
)

n_copies = 4
random_seed = 0

options = prm.PRMOptions(max_vertices=1e4, neighbor_k=12, neighbor_radius=5e0, neighbor_mode="k")

meshcat = StartMeshcat()

limits = [[0, 20], [0, 20]]
params = path_planning_2d.SetupParams(2, limits, 120, 0.75, 0)
diagram, CollisionChecker = path_planning_2d.build_env(meshcat, params, n_copies=n_copies)

diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.plant().GetMyContextFromRoot(diagram_context)
diagram.ForcedPublish(diagram_context)

limits_lower = [limits[0][0], limits[1][0], 0] * n_copies
limits_upper = [limits[0][1], limits[1][1], 0] * n_copies

cspace_dim = 3 * n_copies
symmetry_indices = list(range(2, 3*n_copies, 3))

c_free_volume = 1
c_free_volume *= limits[0][1] - limits[0][0]
c_free_volume *= limits[1][1] - limits[1][0]
c_free_volume *= asymptotic.s1_volume()
c_free_volume = c_free_volume ** n_copies
print("Symmetry-Aware PRM* Minimum Radius:", asymptotic.radius_prm(3, c_free_volume / 3))
print("Symmetry-Unaware PRM* Minimum Radius:", asymptotic.radius_prm(3, c_free_volume))
print("KNN-PRM* Minimum k:", asymptotic.knn_prm(3))

options.neighbor_radius = asymptotic.radius_prm(3, c_free_volume)
options.neighbor_k = asymptotic.knn_prm(3)

G = symmetry.CyclicGroupSO2(2)
Sampler = imacs.SO2SampleUniform(G, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric = imacs.SO2DistanceSqMultiple(G, cspace_dim, symmetry_indices)
Interpolator = imacs.SO2InterpolateMultiple(G, cspace_dim, symmetry_indices)
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionChecker, options)

np.random.seed(random_seed)
roadmap.build(verbose=True)

# Visualize a random plan

while True:
    q0 = Sampler(1)[0]
    if CollisionChecker.CheckConfigCollisionFree(q0):
        break
while True:
    q1 = Sampler(1)[0]
    if CollisionChecker.CheckConfigCollisionFree(q1):
        break

assert CollisionChecker.CheckConfigCollisionFree(q0)
assert CollisionChecker.CheckConfigCollisionFree(q1)

diagram.plant().SetPositions(plant_context, q0)
diagram.ForcedPublish(diagram_context)

path = roadmap.plan(q0, q1)
path = imacs.UnwrapToContinuousPath2dMultiple(G, path, symmetry_indices)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

print("SE(2)/G path length:", Metric.path_length(path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj1 = CompositeTrajectory.AlignAndConcatenate(segments)

dt = traj1.end_time() - traj1.start_time()
dt /= 400
for t in np.linspace(traj1.start_time(), traj1.end_time(), 400):
    diagram.plant().SetPositions(plant_context, traj1.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Now compare to the plan without symmetries

G = symmetry.CyclicGroupSO2(1)
Sampler = imacs.SO2SampleUniform(G, cspace_dim, symmetry_indices, limits_lower, limits_upper)
Metric = imacs.SO2DistanceSqMultiple(G, cspace_dim, symmetry_indices)
Interpolator = imacs.SO2InterpolateMultiple(G, cspace_dim, symmetry_indices)
roadmap = prm.PRM(Sampler, Metric, Interpolator, CollisionChecker, options)

np.random.seed(0)
roadmap.build(verbose=True)

fname = "check_prm_2d_baseline.pkl"
roadmap.save(fname)
roadmap = prm.PRM.load(fname, CollisionChecker)

path = roadmap.plan(q0, q1)
path = imacs.UnwrapToContinuousPath2dMultiple(G, path, symmetry_indices)

for i in range(1, len(path)):
    # print(path[i-1][2], path[i][2])
    assert np.abs(path[i-1][2] - path[i][2]) <= np.pi

print("Baseline path length:", Metric.path_length(path))
t_scaling = 1/4
times = [t_scaling * np.sqrt(Metric(path[i-1], path[i])[0]) for i in range(1, len(path))]
segments = [PiecewisePolynomial.FirstOrderHold([0, times[i-1]], np.array([path[i-1], path[i]]).T) for i in range(1, len(path))]
traj2 = CompositeTrajectory.AlignAndConcatenate(segments)

dt = traj1.end_time() - traj1.start_time()
dt /= 400
for t in np.linspace(traj2.start_time(), traj2.end_time(), 400):
    diagram.plant().SetPositions(plant_context, traj2.value(t).flatten())
    diagram.ForcedPublish(diagram_context)
    time.sleep(dt)

# Alternate visualizing each one

while True:
    for traj, name in zip([traj1, traj2], ["Symmetry-Aware PRM*", "Symmetry-Unaware PRM*"]):
        print(name)
        time.sleep(3)
        dt = traj.end_time() - traj.start_time()
        dt /= 400
        for t in np.linspace(traj.start_time(), traj.end_time(), 400):
            diagram.plant().SetPositions(plant_context, traj.value(t).flatten())
            diagram.ForcedPublish(diagram_context)
            time.sleep(dt)