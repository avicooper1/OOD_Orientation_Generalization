#!/bin/bash
#SBATCH --array=[0]
#SBATCH --time 20:00:00
#SBATCH -n 1
#SBATCH --mem=25G

#For training other models
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity&11GB
#SBATCH --output=/home/avic/om5/training_logs/R_%A_%a.out
#SBATCH --error=/home/avic/om5/training_logs/R_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --partition=normal

hostname
nvidia-smi
module load openmind/singularity/3.5.0
singularity exec --nv -B /om2:/om2 -B /om5:/om5 /home/avic/om5/pytorch.simg \
python -u /home/avic/Rotation-Generalization/train/run.py ${SLURM_ARRAY_TASK_ID}
