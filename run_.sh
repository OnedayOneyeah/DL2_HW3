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

# srun python main_.py --aug FIXMATCH --model resnet34 --pretrained --lr 0.005 --weight_decay 5e-4
# srun python main_.py --aug FIXMATCH --model resnet34 --pretrained --lr 0.01 --weight_decay 5e-4
#  srun python main_.py --aug FIXMATCH --model resnet34 --pretrained --lr 0.01 --weight_decay 5e-4
srun python main_.py --aug FIXMATCH --model resnet18 --pretrained --lr 0.01 --weight_decay 5e-4 --aug_sev both
# srun python main_.py --aug FIXMATCH --model resnet34 --pretrained --lr 0.01 --weight_decay 5e-4
# srun python main_.py --aug FIXMATCH --model resnet18 --pretrained --aug_sev strong
# srun python main_.py --aug FIXMATCH --model resnet18 --pretrained --aug_sev weak
# srun python main_.py --aug Normalize --model resnet18 --pretrained

