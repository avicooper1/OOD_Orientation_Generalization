#!/bin/bash
#SBATCH --array=[3,4,9,22,23]
#SBATCH -t 07:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=2G
#SBATCH --exclude node109,node112,node113,node106,node102,node100
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --output=/home/avic/om2/logs/render/R%A_%a.out
#SBATCH --error=/home/avic/om2/logs/render/R%A_%a.err
#SBATCH --partition=normal

hostname
~/om2/software/blender/blender -b -noaudio -P ~/OOD_Orientation_Generalization/render/render.py ~/OOD_Orientation_Generalization ~/om2/OODOG/ 32 lamp ${SLURM_ARRAY_TASK_ID}
