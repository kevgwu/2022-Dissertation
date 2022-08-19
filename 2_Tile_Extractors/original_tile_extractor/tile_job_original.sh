#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N tiling
#$ -cwd
#$ -l h_rt=00:29:00
#$ -l h_vmem=32G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
source activate rasterio

# Run the program
python tile_extractor_boa.py
