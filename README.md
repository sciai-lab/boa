# BOA

## Installation
Conda
```bash
conda env create -f environment.yaml
conda activate boa
pip install -e scdp/ sciai-dft/ .
```
UV
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install -e scdp/ sciai-dft/ .
```

## Environment Variables
```bash
export BOA_DATA="/export/scratch/mklockow/charge_density_lmdb"
export BOA_MODELS="."
```
Quick fix for project root
```bash
touch .env
```
