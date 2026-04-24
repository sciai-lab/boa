# BOA
This repository contains the code to reproduce the results in the ICLR paper [A Function-Centric Graph Neural Network Approach For Predicting Electron Densities](https://openreview.net/pdf?id=HDdkFjFEZd).

## Installation
For environment management, we use UV:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install -e scdp/ -e sciai-dft/ -e .
```

## Environment Variables
Fill .env with following lines to set environment variables:
```bash
BOA_DATA=""
BOA_MODELS=""
```

## Training

```bash
python boa/train.py experiment=<your_experiment>
```

## Testing

```bash
python boa/test.py eval=<your_eval>
```