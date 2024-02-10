#!/bin/bash
#SBATCH --job-name=benchmark_rot_mnist_cano
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --reservation=ubuntu1804
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-47
#SBATCH --output=rotmnist_sweep_output/experiment-%A.%a.out

module load anaconda/3
equiadapt # equivalent to conda activate equiadapt

# To create a sweep use the corresp
# wandb sweep --project equiadapt -e symmetry_group <sweep-name>.yaml

# To run the agent of the created sweep
wandb agent symmetry_group/equiadapt/1mb3icnk
