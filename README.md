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
