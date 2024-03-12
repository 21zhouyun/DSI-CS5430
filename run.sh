#!/bin/sh
#SBATH --job-name=cs5430-dsi
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhouyun@comp.nus.edu.sg
#SBATCH --gpus=v100:2
#SBATCH --partition=long

srun bash init.sh
srun python train.py --train_data $1 --validation_data $2 --output_dir $3
