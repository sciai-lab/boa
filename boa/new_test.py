import logging
import sys
import traceback
from pathlib import Path
from typing import List

import hydra
import omegaconf
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import boa.utils.omegaconf_resolvers  # noqa E402
from mldft.utils.log_utils.config_in_tensorboard import dict_to_tree  # noqa E402


def run(cfg: DictConfig):
    import time

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch import Callback, seed_everything
    from lightning.pytorch.loggers import TensorBoardLogger

    from boa.model.module import ChgLightningModule
    from boa.train import build_callbacks
    from scdp.common.system import log_hyperparameters  # noqa E402

    torch.multiprocessing.set_sharing_strategy("file_system")
    pylogger = logging.getLogger(__name__)
    seed_everything(42)

    ckpt_path = Path(cfg.ckpt_path)
    # pylint: disable=E1120
    pylogger.info(f"loaded checkpoint: {ckpt_path}")
    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt_path).to("cuda")
    model.eval()
    model.ema.copy_to(model.parameters())

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    callbacks: List[Callback] = build_callbacks(cfg.callbacks)
    logger = TensorBoardLogger(**cfg.logger.tensorboard)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    pylogger.info("Starting testing.")
    curr_time = time.time()
    trainer.test(model=model, datamodule=datamodule)
    # save the nmape results ckpt_path / f'nmape_{tag}.txt'
    # torch.save(all_nmapes, ckpt_path / f"nmape_{tag}.pt")

    elapsed_time = time.time() - curr_time
    pylogger.info(f"elapsed time: {elapsed_time:.2f} seconds.")


@hydra.main(
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="test",
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
