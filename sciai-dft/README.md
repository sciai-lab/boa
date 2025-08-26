<div align="center">

# SCIAI-DFT

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.5-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

</div>

For the latest version of this caode please refer to the [repository](https://github.com/sciai-lab/structures25).

## Installation

#### Install using Conda/Mamba/Micromamba (recommended)

For conda or mamba replace `micromamba` with `conda` or `mamba` below.
If you want to create the environment with CPU support only, you can replace
`environment.yaml` with `environment_cpu.yaml`.

```bash
micromamba env create -f environment.yaml  # create mamba environment
micromamba activate mldft                  # activate environment
pip install -e .                           # install as an editable package
pip install -e tensorframes                # install tensorframes
```

#### Install using Pip

```bash
pip install -r requirements.txt -e .       # install requirements and package
pip install -e tensorframes                # install tensorframes
```

#### Environment variables

Before running the code you need to set the two following environment variables `DFT_DATA`, the path
where the data should be stored and `DFT_MODELS` which is the path where the training runs
including model checkpoints, logs and tensorboard files should be stored. You can set them in your
`.bashrc` or `.zshrc`

```bash
export DFT_DATA="/path/to/data"
export DFT_MODELS="/path/to/models"
```

## Usage

This is a general usage manual. To reproduce results from our paper see [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md).

We use [hydra](https://hydra.cc/docs/intro/) to manage configurations. The main configuration files are located in `configs/`.

#### Data generation

1. (Optional) Create your own dataset class or use the MISC dataset and provide xyz files to set which molecules should be generated.

2. (Optional) Create a config file in `configs/datagen/dataset/`

3. Run Kohn-Sham DFT on the dataset and create `.chk` files in `$DFT_DATA/dataset/kohn_sham`:

   Example: `mldft_ks dataset=<your_dataset_config_name> n_molecules=1000 start_idx=0`

4. Based on the Kohn-Sham result, do density fitting, compute energy and gradients and save as labels for the machine learning model in `$DFT_DATA/dataset/labels`:

   Example: `mldft_labelgen dataset=<your_dataset_config_name> n_molecules=-1`

5. Split the file into train, validation and test dataset using`mldft/utils/create_dataset_splits.py`.

   Example: `python mldft/utils/create_dataset_splits.py <dataset_name>`

6. Create a train data config in `configs/ml/data` to link to the dataset, important are dataset_name and the right setting of atom types in the dataset.

7. Transform into a basis (to reduce dataloading computations during training), for `Graphformer` models use `local_frames_global_natrep`.

   Example: `python mldft/datagen/transform_dataset.py data=<your_train_data_config_name> data/transforms=local_frames_global_natrep`

8. Compute dataset statistics, important is to compute them for the transformation and the energy target that you want to use.

   Example: `python mldft/ml/compute_dataset_statistics.py data=<your_dataset_config_name>`

Now you can start [Training](#Training)

#### Training

Training can be run with:

```bash
python mldft/ml/train.py data=<train_data_config> model=<model_config>
```

Two important settings are

- `data/transforms`: This determines whether the data has been pre-transformed. The default is `local_frames_global_natrep` which means that both *local frames* and *global natural reparametrization* has been applied.
- `data.target_key`: The target you are training. The default is `kin_plus_xc` which means you train on the total kinetic energy and exchange-correlation energy and their gradient. Alternatives are `kin_minus_apbe` which is a delta learning approach to the kinetic energy obtained from the APBE kinetic energy functional and `tot` which means you are training on the total electronic energy.

#### Density Optimization

To run density optimization on a dataset in our format, you can run the following command:

```bash
python mldft/ofdft/run_density_optimization run_path=<path_to_ml_model> \
    n_molecules=<number_of_molecules> device=<device> initialization=<initialization> num_devices=<num_devices>
```

- `path_to_model` is the path of to the model relative to `DFT_MODELS`
- `n_molecules` the number of molecules which should be computed
- `device` the device on which the computation should run, e.g. `cuda`, `cpu`, ...
- `initialization` the initialization to use either `sad`, `minao` or `hückel`, the `sad` initialization requires appropriate dataset statistics.

By default this will run on the validation set of the dataset the model was trained on, but you can overwrite `split_file_path` to use another split file and `split` to toggle between `train`, `val` and `test` splits of the dataset.
Results are plotted in the files `density_optimization.pdf` and `density_optimization_summary.pdf`.

## Additional Info

#### Build documentation

```bash
make docs
# or to build from scratch:
make docs-clean
```

#### Template

For more details about the template visit: https://github.com/ashleve/lightning-hydra-template

## Third-party licenses

This code adapts code from the following third party libraries:

- [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
- [equiformer_v2](https://github.com/atomicarchitects/equiformer_v2)

These are distributed under the MIT license which can be found in the [license file](LICENSE).
