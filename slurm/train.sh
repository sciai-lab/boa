#!/bin/bash
#SBATCH --partition=gpu_h100        # Partition (choose from devel, single, cpu-mult, gpu-multi; more info at https://wiki.bwhpc.de/e/Helix/Slurm)
#SBATCH --nodes=1                   # Number of nodes, must match trainer.nodes
#SBATCH --gres=gpu:1                # Number of GPUs, must match trainer.devices
#SBATCH --ntasks-per-node=1         # Number of tasks per node, must match trainer.devices
#SBATCH --cpus-per-task=16           # Number of CPUs per task, must match data.datamodule.num_workers
#SBATCH --signal=SIGUSR1@300        # Signal to send to the job on timeout. This is handled by lightning and will save a checkpoint and resubmit the job to continue from that checkpoint.
#SBATCH --time=72:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --open-mode=append          # Append to the output file (if it already exists). This is useful for resuming from checkpoints.
#SBATCH --export=NONE

source ~/.bashrc
module purge

cd ~/boa/
source .venv/bin/activate
srun python boa/train.py trainer=slurm

