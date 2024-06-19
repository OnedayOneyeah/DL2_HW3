#!/bin/bash

#SBATCH --job-name=cub200_herbs_basline     # Submit a job named "example"
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:2                 # Using 1 gpu
#SBATCH --time=0-12:00:00            # 1 hour time limit
#SBATCH --mem=10000MB                # Using 10GB CPU Memory
#SBATCH --cpus-per-task=4            # Using 4 maximum processor
#SBATCH --output=./S-%x.%j.out       # Make a log file

eval "$(conda shell.bash hook)"
conda activate hw3

srun python main.py --aug HERBS --model resnet50

