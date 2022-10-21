#!/bin/bash
#SBATCH --array=[0-49]
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --exclude node109,node112,node113,node106,node102
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --output=/home/avic/om2/logs/finalize_render/R%A_%a.out
#SBATCH --error=/home/avic/om2/logs/finalize_render/R%A_%a.err
#SBATCH --partition=normal

hostname
python3 /home/avic/OOD_Orientation_Generalization/render/finalize_render.py /home/avic/om2/OODOG 32 plane -sd ${SLURM_ARRAY_TASK_ID}
