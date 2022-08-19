#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N road123
#$ -cwd
#$ -pe sharedmem 4
#$ -l h_rt=00:29:00
#$ -l h_vmem=64G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Python
module load anaconda
source activate roadenv

# Run the program
python road_binary_raster.py
