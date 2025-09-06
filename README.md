# BOA

## Installation
Conda
```bash
conda env create -f environment.yaml
conda activate boa
pip install -e scdp/ -e sciai-dft/ -e .
```
UV
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install -e scdp/ -e sciai-dft/ -e .
```

## Environment Variables
Fill .env with following lines to set environment variables:
```bash
BOA_DATA="/export/scratch/mklockow/charge_density_lmdb"
BOA_MODELS="."
```
## Hydra SLURM Job Submission

To submit a job to SLURM you need to set -m to switch on the multi run which is necessary for submitit.

```bash
python boa/train.py -m experiment=qm9_vasp_small_slurm
```

```bash
python boa/train.py experiment=qm9_vasp_small_slurm,qm9_vasp_small_noabs_slurm
```

Debugging
```bash
python boa/train.py experiment=qm9_vasp_small_slurm hydra/launcher=submit_slurm_no_copy hydra.launcher.timeout_min=4 initial_guess_pre_training_steps=2 hydra.launcher.partition="gpu_a100_short" -m
```
plus setting sigusr lower in config.