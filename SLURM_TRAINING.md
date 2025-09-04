# SLURM Training with Hydra Submitit

This configuration allows you to submit training jobs to SLURM using Hydra's submitit launcher plugin instead of manual SLURM scripts.

## Installation

First, install the required dependencies:

```bash
pip install hydra-submitit-launcher
```

## Usage

### Basic SLURM Job Submission

To submit a job to SLURM using the configuration from your original `train.sh` script:

```bash
python boa/train.py hydra=slurm trainer=slurm
```

This will:
- Use the gpu_h100 partition
- Request 1 node with 1 GPU
- Use 16 CPUs per task
- Set a 72-hour time limit
- Configure signal handling for timeouts
- Set up the Python environment as specified in your original script

### Customizing SLURM Parameters

You can override any SLURM parameter from the command line:

```bash
# Change partition and time limit
python boa/train.py hydra=slurm trainer=slurm hydra.launcher.partition=gpu_a100 hydra.launcher.time="24:00:00"

# Use multiple GPUs
python boa/train.py hydra=slurm trainer=slurm hydra.launcher.gpus_per_node=4 trainer.devices=4

# Change number of CPUs
python boa/train.py hydra=slurm trainer=slurm hydra.launcher.cpus_per_task=32
```

### Running Hyperparameter Sweeps

You can also run parameter sweeps on SLURM:

```bash
python boa/train.py hydra=slurm trainer=slurm -m model.lr=0.001,0.01,0.1
```

This will submit separate SLURM jobs for each hyperparameter combination.
