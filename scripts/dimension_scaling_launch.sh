#!/bin/bash

# Usage
# LLsub ./scripts/dimension_scaling_launch.sh -s 48

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
NSIDES=3
NWORLDS=10
NPAIRS=100
MAXDIMENSION=12

echo "[supercloud_launch.sh] Shape: $SHAPE"
echo "[supercloud_launch.sh] Number of Sides: $NSIDES"
echo "[supercloud_launch.sh] Number of Worlds: $NWORLDS"
echo "[supercloud_launch.sh] Number of Pairs Per World: $NPAIRS"
echo "[supercloud_launch.sh] Maximum Dimension: $MAXDIMENSION"

for i in $(seq 3 3 $MAXDIMENSION);
do python3 scripts/statistics_multiple_prm.py --dimension=$i --n_sides=$NSIDES --n_worlds=$NWORLDS --n_pairs_per_world=$NPAIRS
done
