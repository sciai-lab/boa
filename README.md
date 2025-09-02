# BOA

## Installation
#### Requirements
```bash
conda env create -f environment.yaml
```
```bash
pip install -r requirements.txt
```
```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

#### Local Packages
```bash
pip install -e scdp/ sciai-dft/ .
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
