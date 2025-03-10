import sys
sys.path.append("..")

import numpy as np
import pathlib
import time
from scipy.stats import special_ortho_group

import src.symmetry
import src.util

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

directives_str = """directives:
- add_model:
    name: dodecahedron
    file: package://symmetries/models/dodecahedron.sdf
"""
G = src.symmetry.IcosahedralGroup()

meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

parser = Parser(plant)
parser.package_map().Add("symmetries", src.util.repo_dir())
directives = LoadModelDirectivesFromString(directives_str)
models = ProcessModelDirectives(directives, plant, parser)

plant.Finalize()

# meshcat_visual_params = MeshcatVisualizerParams()
# meshcat_visual_params.delete_on_initialization_event = False
# meshcat_visual_params.role = Role.kIllustration
# meshcat_visual_params.prefix = "visual"
# meshcat_visual = MeshcatVisualizer.AddToBuilder(
#   builder, scene_graph, meshcat, meshcat_visual_params)

meshcat_collision_params = MeshcatVisualizerParams()
meshcat_collision_params.delete_on_initialization_event = False
meshcat_collision_params.role = Role.kProximity
meshcat_collision_params.prefix = "collision"
meshcat_collision_params.visible_by_default = True
meshcat_collision = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, meshcat_collision_params)

# meshcat_visual.Delete()
meshcat_collision.Delete()

diagram = builder.Build()

button = meshcat.AddButton("Next")
num_clicks = 0
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)

mat = np.eye(3)
mat = special_ortho_group.rvs(3)
tfs_so3 = G.orbit(mat)

for tf_so3 in tfs_so3:
    q = RotationMatrix(tf_so3).ToQuaternion().wxyz()
    positions = np.hstack((q, np.zeros(3)))
    plant.SetPositions(plant_context, positions)

    diagram.ForcedPublish(diagram_context)

    while meshcat.GetButtonClicks("Next") == num_clicks:
        time.sleep(0.05)

    num_clicks += 1

meshcat.DeleteButton("Next")

print("Done! The object should not have appeared to move at all.")