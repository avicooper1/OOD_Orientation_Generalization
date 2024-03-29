#!/bin/bash
#SBATCH --array=[0,1,2,3]
#SBATCH --time 01:00:00
#SBATCH -n 4
#SBATCH --mem=8G

#For training other models
#SBATCH --gres=gpu:1
#SBATCH --output=/home/avic/om5/activation_collection_logs/R_%A_%a.out
#SBATCH --error=/home/avic/om5/activation_collection_logs/R_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --partition=normal

hostname
nvidia-smi
module load openmind/singularity/3.5.0
singularity exec --nv -B /om2:/om2 -B /om5:/om5 /home/avic/om5/pytorch.simg python -u /home/avic/OOD_Orientation_Generalization/analysis/collect_activations.py ${SLURM_ARRAY_TASK_ID}