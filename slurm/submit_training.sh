#!/bin/bash
#SBATCH --time 05:00:00
#SBATCH --array=[963,991,1130,1137,1141,1151,1165,1167,1170,1171,1173,1174,1175,1177,1178,1179]
#SBATCH -c 4
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-A100
#SBATCH --output=/home/avic/om2/logs/train/R_%A_%a.out
#SBATCH --error=/home/avic/om2/logs/train/R_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --partition=normal

hostname
nvidia-smi
python3 /home/avic/OOD_Orientation_Generalization/train/run.py /home/avic/OOD_Orientation_Generalization/ /home/avic/om2/OODOG/ ${SLURM_ARRAY_TASK_ID}
