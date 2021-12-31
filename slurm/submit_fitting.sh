#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=END
#SBATCH --mail-user=avic@mit.edu
#SBATCH --output=/home/avic/om5/fits_logs/R%A_%a.out
#SBATCH --error=/home/avic/om5/fits_logs/R%A_%a.err
#SBATCH --partition=normal

hostname
python3 /home/avic/Rotation-Generalization/analysis/run_fitting2.py 1
#python3 /home/avic/Rotation-Generalization/analysis/run_fitting2.py 2