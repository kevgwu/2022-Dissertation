#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N model2_2
#$ -cwd
#$ -pe gpu-titanx 1
#$ -l h_rt=48:00:00
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
source activate gpu-torch

# Run the program
for i in {1..500}; do python Model2_test.py "$i"; done
