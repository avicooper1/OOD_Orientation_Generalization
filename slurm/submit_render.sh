#!/bin/bash
#SBATCH --array=[1745,1748,1966,1971,1976,1978,1979,1981,1982,1983,1984,1985,1992,1995,1996,1999]
#SBATCH -t 06:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --exclude node093,node094,node097
#SBATCH --constraint=any-gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --output=/home/avic/om5/render_logs/R%A_%a.out
#SBATCH --error=/home/avic/om5/render_logs/R%A_%a.err
#SBATCH --partition=normal

hostname
nvidia-smi
~/blender -b -noaudio -P /home/avic/Rotation-Generalization/render/render.py ${SLURM_ARRAY_TASK_ID}
