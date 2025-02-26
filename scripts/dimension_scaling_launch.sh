#!/bin/bash

# Usage
# LLsub ./scripts/dimension_scaling_launch.sh -s 48

# Initialize and Load Modules
echo "[dimension_scaling_launch.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/Python-ML-2025a

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`

echo "[dimension_scaling_launch.sh] Date: $DATE"
echo "[dimension_scaling_launch.sh] Time: $TIME"

SHAPE=polygon
NSIDES=2
NWORLDS=10
NPAIRS=100
MAXDIMENSION=30
NVERTICES=10000

echo "[dimension_scaling_launch.sh] Shape: $SHAPE"
echo "[dimension_scaling_launch.sh] Number of Sides: $NSIDES"
echo "[dimension_scaling_launch.sh] Number of Worlds: $NWORLDS"
echo "[dimension_scaling_launch.sh] Number of Pairs Per World: $NPAIRS"
echo "[dimension_scaling_launch.sh] Maximum Dimension: $MAXDIMENSION"
echo "[dimension_scaling_launch.sh] Number of Vertices: $NVERTICES"

#for i in $(seq 3 3 $MAXDIMENSION);
# do python3 scripts/statistics_multiple_prm.py --dimension=$i --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS --n_vertices=$NVERTICES
#do python3 scripts/statistics_multiple_rrt.py --dimension=$i --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS --rrt_nodes_max=$NVERTICES --step_size=5.0 --birrt
#done

python3 scripts/statistics_multiple_rrt.py --dimension=$MAXDIMENSION --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS --rrt_nodes_max=$NVERTICES --step_size=5.0 --birrt
