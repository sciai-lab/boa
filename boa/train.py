import json
import logging
from pathlib import Path
from typing import List

import hydra
import lightning.pytorch as pl
import omegaconf
import rich
import rootutils
import torch
from lightning.pytorch import Callback, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# NOTE: disable slurm detection of lightning
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, ListConfig, open_dict

from mldft.utils.log_utils.config_in_tensorboard import dict_to_tree

# this import registers custom omegaconf resolvers
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import boa.utils.omegaconf_resolvers  # noqa E402
from scdp.common.system import PROJECT_ROOT, log_hyperparameters  # noqa E402

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/config.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/config.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

torch.multiprocessing.set_sharing_strategy("file_system")
SLURMEnvironment.detect = lambda: False
pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    if cfg.deterministic:
        seed_everything(cfg.seed)

    # print the resolved config using Rich library
    if cfg.get("print_config"):
        rich.print(dict_to_tree(cfg, guide_style="dim"))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.data.datamodule['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")

    metadata = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(
            f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>"
        )

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.model['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        train=cfg,
        _recursive_=False,
        metadata=metadata,
    )

    callbacks: List[Callback] = build_callbacks(cfg.callbacks)

    storage_dir: str = cfg.paths.output_dir

    if "wandb" in cfg.logger:
        pylogger.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logger.wandb
        logger = WandbLogger(**wandb_config)
        pylogger.info(f"W&B is now watching <{cfg.logger.wandb_watch.log}>!")
        logger.watch(
            model,
            log=cfg.logger.wandb_watch.log,
            log_freq=cfg.logger.wandb_watch.log_freq,
        )
    else:
        logger = TensorBoardLogger(**cfg.logger.tensorboard)
        pylogger.info(f"TensorBoard Logger logs into <{cfg.logger.tensorboard.save_dir}>.")

    ckpt = None

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer,
    )

    # save the config yaml file.
    yaml_conf: str = omegaconf.OmegaConf.to_yaml(cfg)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    log_hyperparameters(cfg, model, trainer)
    with open(Path(storage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)

    assert not (cfg.ckpt_path and cfg.weight_ckpt_path), (
        "Only one of `ckpt_path` or `weight_ckpt_path` can be provided."
    )
    if cfg.ckpt_path:
        ckpt = cfg.ckpt_path
        pylogger.info(f"Using checkpoint: {ckpt}")
    elif cfg.weight_ckpt_path:
        # load state dict of model
        state = torch.load(cfg.weight_ckpt_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(state, strict=False)
        pylogger.info(f"Loaded model weights from: {cfg.weight_ckpt_path}")

    if not cfg.ckpt_path and not cfg.weight_ckpt_path:
        if cfg.initial_guess_pre_training_steps > 0:
            pre_cfg = cfg.copy()
            with open_dict(pre_cfg.model):
                pre_cfg.model.net = pre_cfg.model.net.initial_guess_module
                if "pre_training_overrides" in cfg:
                    pre_cfg = omegaconf.OmegaConf.merge(pre_cfg, cfg.pre_training_overrides)
                    # pretty print the new pre_cfg
                    pylogger.info(
                        f"Pre-training overrides:\n{omegaconf.OmegaConf.to_yaml(cfg.pre_training_overrides)}"
                    )
                if "pre_training_replacements" in cfg:
                    for key, value in cfg.pre_training_replacements.items():
                        pre_cfg[key] = value
                    pylogger.info(
                        f"Pre-training config replacements:\n{omegaconf.OmegaConf.to_yaml(cfg.pre_training_replacements)}"
                    )

        pre_datamodule = hydra.utils.instantiate(pre_cfg.data.datamodule, _recursive_=False)
        pre_datamodule.setup(stage="fit")

        pre_metadata = getattr(pre_datamodule, "metadata", None)
        if pre_metadata is None:
            pylogger.warning(
                f"No 'metadata' attribute found in datamodule <{pre_datamodule.__class__.__name__}>"
            )

        pre_model = hydra.utils.instantiate(
            pre_cfg.model,
            train=pre_cfg,
            _recursive_=False,
            metadata=pre_metadata,
        )

        pre_cfg.logger.tensorboard.save_dir = (
            pre_cfg.logger.tensorboard.save_dir[:-1] + "_pre_training/"
        )
        pre_logger = TensorBoardLogger(**pre_cfg.logger.tensorboard)

        pre_trainer = pl.Trainer(
            logger=pre_logger,
            **pre_cfg.trainer,
        )
        pre_trainer.fit(
            pre_model, pre_datamodule.train_dataloader(), pre_datamodule.val_dataloader()
        )

        model.model.initial_guess_module.load_state_dict(pre_model.model.state_dict())

    pylogger.info("starting training.")
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt)

    if (
        datamodule.test_dataset is not None
        and trainer.checkpoint_callback.best_model_path is not None
    ):
        pylogger.info("starting testing.")
        trainer.test(dataloaders=[datamodule.test_dataloader()])


@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="train", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    # pylint: disable=E1120
    main()
