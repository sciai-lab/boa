#!/bin/bash
#SBATCH --partition=gpu_a100_il,gpu_h100_il,gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --signal=SIGUSR1@180
#SBATCH --time=48:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/hd_ai306-dft_data/models/sbatch_output/%x.%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/hd_ai306-dft_data/models/sbatch_output/%x.%j.out
#SBATCH --open-mode=append
#SBATCH --export=NONE

# source ~/.bashrc
module purge

export BOA_DATA="/pfs/work9/workspace/scratch/hd_ai306-dft_data/data"
export BOA_MODELS="/pfs/work9/workspace/scratch/hd_ai306-dft_data/models"
cd ~/boa/
source .venv/bin/activate
srun --export=ALL python -u boa/train.py experiment=qm9_vasp_small_noradcor trainer=slurm
