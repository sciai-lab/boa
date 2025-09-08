import logging
import sys
import traceback
from pathlib import Path
from random import randint
from typing import List

import hydra
import omegaconf
import rich
import rootutils
from omegaconf import DictConfig, ListConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import boa.utils.omegaconf_resolvers  # noqa E402
from mldft.utils.log_utils.config_in_tensorboard import dict_to_tree  # noqa E402

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args) -> List:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List = list(args)

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
    import json

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

    from scdp.common.system import log_hyperparameters  # noqa E402

    torch.multiprocessing.set_sharing_strategy("file_system")
    pylogger = logging.getLogger(__name__)
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)
    else:
        seed = randint(0, 2**32 - 1)
        seed_everything(seed, workers=True)
        with open_dict(cfg):
            cfg.seed = seed

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

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    yaml_conf: str = omegaconf.OmegaConf.to_yaml(cfg)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    (Path(storage_dir) / "config.yaml").write_text(yaml_conf)
    log_hyperparameters(cfg, model, trainer)
    with open(Path(storage_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)

    assert not (cfg.ckpt_path and cfg.weight_ckpt_path), (
        "Only one of `ckpt_path` or `weight_ckpt_path` can be provided."
    )

    # Check whether pretraining was finished before
    pretraining_done_marker = Path(storage_dir) / ".pretraining_completed"

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

    # Only do pretraining if we're not resuming from a checkpoint and pretraining hasn't been completed
    should_do_pretraining = (
        not cfg.ckpt_path
        and not cfg.weight_ckpt_path
        and not pretraining_done_marker.exists()
        and cfg.initial_guess_pre_training_steps > 0
    )

    if should_do_pretraining:
        pylogger.info("Starting initial guess pre-training...")
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

        pre_trainer = hydra.utils.instantiate(pre_cfg.trainer, logger=pre_logger)
        pre_trainer.fit(
            pre_model, pre_datamodule.train_dataloader(), pre_datamodule.val_dataloader()
        )

        model.model.initial_guess_module.load_state_dict(pre_model.model.state_dict())

        # Mark pretraining as completed
        pretraining_done_marker.touch()
        pylogger.info(f"Pre-training completed. Marker saved at: {pretraining_done_marker}")
    else:
        if pretraining_done_marker.exists():
            pylogger.info("Skipping pre-training (already completed in previous run)")
        elif cfg.ckpt_path or cfg.weight_ckpt_path:
            pylogger.info("Skipping pre-training (checkpoint/weights provided)")
        else:
            pylogger.info("Skipping pre-training (initial_guess_pre_training_steps is 0)")

    pylogger.info("starting training.")
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=ckpt)

    # save best model path if available
    if trainer.checkpoint_callback.best_model_path:
        (Path(storage_dir) / "best_model_path.txt").write_text(
            trainer.checkpoint_callback.best_model_path
        )
        pylogger.info(
            f"Best model path: {trainer.checkpoint_callback.best_model_path} (score: {trainer.checkpoint_callback.best_model_score})"
        )

    if (
        datamodule.test_dataset is not None
        and trainer.checkpoint_callback.best_model_path is not None
    ):
        pylogger.info("starting testing.")
        trainer.test(dataloaders=[datamodule.test_dataloader()])


@hydra.main(
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="train",
    version_base="1.3",
)
def main(cfg: omegaconf.DictConfig):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    # pylint: disable=E1120
    main()
