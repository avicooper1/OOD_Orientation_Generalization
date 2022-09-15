#!/bin/bash
#SBATCH --array=0-199
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1G
#SBATCH --exclude node109,node112,node113,node106,node102
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --output=/home/avic/om2/logs/render/R%A_%a.out
#SBATCH --error=/home/avic/om2/logs/render/R%A_%a.err
#SBATCH --partition=normal

hostname
nvidia-smi
~/om2/software/blender/blender -b -noaudio -P ~/OOD_Orientation_Generalization/render/render.py /home/avic/om2/OODOG 32 ${SLURM_ARRAY_TASK_ID}
