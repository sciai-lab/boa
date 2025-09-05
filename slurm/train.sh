#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --signal=SIGUSR1@300
#SBATCH --time=72:00:00
#SBATCH --open-mode=append
#SBATCH --export=NONE

source ~/.bashrc
module purge

# Ensure extraction directory exists
mkdir -p $TMPDIR
# Decompress and extract to $TMPDIR
zstd --decompress -T0 /pfs/work9/workspace/scratch/hd_ai306-dft_data/data/qm9_vasp.tar.zst | tar --extract --file - -C $TMPDIR/

export BOA_DATA=$TMPDIR
export BOA_MODELS=/pfs/work9/workspace/scratch/hd_ai306-dft_data/models
cd ~/boa/
source .venv/bin/activate
srun python boa/train.py experiment=qm9_vasp_small trainer=slurm
