#!/bin/bash

# Usage
# LLsub ./scripts/supercloud_launch.sh -s 48

# Initialize and Load Modules
echo "[supercloud_launch.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/Python-ML-2025a

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`

echo "[supercloud_launch.sh] Date: $DATE"
echo "[supercloud_launch.sh] Time: $TIME"

SHAPE=polygon
NSIDES=2
NWORLDS=10
NPAIRS=100

echo "[supercloud_launch.sh] Shape: $SHAPE"
echo "[supercloud_launch.sh] Number of Sides: $NSIDES"
echo "[supercloud_launch.sh] Number of Worlds: $NWORLDS"
echo "[supercloud_launch.sh] Number of Pairs Per World: $NPAIRS"

python3 scripts/statistics_prm.py --shape=$SHAPE --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS
python3 scripts/statistics_rrt.py --shape=$SHAPE --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS