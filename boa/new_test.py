import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List

import hydra
import omegaconf
import rootutils
from omegaconf import DictConfig, ListConfig

from mldft.utils.counter_file import get_and_increment_counter

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from boa.utils.omegaconf_resolvers import checkpoint_path_to_run_number  # noqa E402
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

    # check if ckpt_path ends with .ckpt
    if not cfg.get("ckpt_path", "").endswith(".ckpt"):
        # check for best_model_path.txt in the ckpt_path directory
        best_model_path_file = Path(cfg.ckpt_path) / "best_model_path.txt"
        if best_model_path_file.exists():
            with open(best_model_path_file, "r") as f:
                best_model_path = f.read().strip()
                ckpt_file = Path(best_model_path).name
                ckpt_path = Path(best_model_path_file).parent / ckpt_file
            pylogger.info(f"Using best model: {ckpt_path}")
        else:
            raise ValueError(
                f"ckpt_path {cfg.ckpt_path} is not a .ckpt file and best_model_path.txt not found."
            )
    else:
        # check if best_model_path.txt exists in the same directory as ckpt_path
        best_model_path = Path(cfg.ckpt_path).parent / "best_model_path.txt"
        if best_model_path.exists():
            # check if the best_model_path is different from the ckpt_path
            with open(best_model_path, "r") as f:
                best_ckpt_path = f.read().strip()
            if best_ckpt_path != cfg.ckpt_path:
                pylogger.info(
                    f"Warning: ckpt_path {cfg.ckpt_path} is not the best model. {best_ckpt_path} is the best model."
                )
        ckpt_path = Path(cfg.ckpt_path)
    # pylint: disable=E1120
    pylogger.info(f"loaded checkpoint: {ckpt_path}")
    model = ChgLightningModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    model.ema.copy_to(model.parameters())
    model.max_n_probe_per_pass = cfg.get("max_n_probe_per_pass", 10000)

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
    # check if there is a list of ckpt_paths
    cfg_full = cfg.copy()
    if not isinstance(cfg.ckpt_path, (ListConfig, list)):
        ckpt_paths = [cfg.ckpt_path]
    else:
        ckpt_paths = cfg.ckpt_path

    output_dirs = []
    for i, ckpt_path in enumerate(ckpt_paths):
        cfg = cfg_full.copy()
        with omegaconf.open_dict(cfg):
            cfg.ckpt_path = ckpt_path
        try:
            # append from_runnumber to hydra.run.dir if ckpt_path is given
            if cfg.get("ckpt_path") is not None:
                run_number = checkpoint_path_to_run_number(cfg.ckpt_path)
                with omegaconf.open_dict(cfg):
                    output_dir = Path(cfg.paths.output_dir)
                    test_run_number = get_and_increment_counter(
                        os.path.join(output_dir.parent, ".run_counter")
                    )
                    output_dir = str(
                        output_dir.parent
                        / (f"{test_run_number:03d}_" + output_dir.name.split("_", 1)[1])
                    )
                    cfg.paths.output_dir = output_dir + f"_from_{run_number}"
                    output_dirs.append(cfg.paths.output_dir)
            run(cfg)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise
    print("All output dirs:")
    print(output_dirs)


if __name__ == "__main__":
    # pylint: disable=E1120
    main()
