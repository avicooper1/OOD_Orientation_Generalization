#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --array=[33]
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity
#SBATCH --output=/home/avic/om2/logs/fit/R_%A_%a.out
#SBATCH --error=/home/avic/om2/logs/fit/R_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --partition=normal

hostname
module load openmind/cuda/11.2
nvidia-smi
python3 /home/avic/OOD_Orientation_Generalization/analysis/generate_results.py /home/avic/OOD_Orientation_Generalization/ /home/avic/om2/OODOG/ --nums ${SLURM_ARRAY_TASK_ID} -fm
