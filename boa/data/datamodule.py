import json
import random
from pathlib import Path
from typing import Dict, Optional

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from boa.data.dataloader import ProbeDataLoader
from boa.data.dataset import LmdbDataset, PyscfDataset
from scdp.common.pyg import DataLoader


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        dataloader_kwargs: DictConfig,
    ):
        super().__init__()
        self.dataset = dataset
        if split_file is not None:
            with open(split_file, "r") as fp:
                self.splits = json.load(fp)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.metadata: Optional[Dict] = None
        self.dataloader_kwargs = dataloader_kwargs

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        self.dataset = hydra.utils.instantiate(self.dataset)
        if isinstance(self.dataset, (LmdbDataset, PyscfDataset)):
            self.train_dataset = Subset(self.dataset, self.splits["train"])
            self.val_dataset = Subset(self.dataset, self.splits["validation"])
            self.test_dataset = Subset(self.dataset, self.splits["test"])
            if (Path(self.dataset.path) / "metadata_full.json").exists():
                with open(self.dataset.path / "metadata_full.json", "r") as fp:
                    self.metadata = json.load(fp)
            else:
                if len(self.train_dataset) > 0:
                    self.metadata = self.get_metadata()
                else:
                    self.metadata = {}
        else:
            # Take 10 % of training as validation
            full_train_dataset = self.dataset(split="train")
            n_val = int(0.1 * len(full_train_dataset))
            n_train = len(full_train_dataset) - n_val
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_train_dataset,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            self.test_dataset = self.dataset(split="test")
            if (
                Path(self.train_dataset.dataset.root)
                / self.train_dataset.dataset.mol_name
                / "metadata_full.json"
            ).exists():
                with open(
                    Path(self.train_dataset.dataset.root)
                    / self.train_dataset.dataset.mol_name
                    / "metadata_full.json",
                    "r",
                ) as fp:
                    self.metadata = json.load(fp)
            else:
                self.metadata = self.get_metadata()

    def get_metadata(self):
        x_sum = 0
        x_2 = 0
        unique_atom_types = set()
        avg_num_neighbors = 0
        print("get metadata.")
        progress = tqdm(total=len(self.train_dataset))
        # if self.train_dataset has n_probe
        if hasattr(self.train_dataset, "n_probe"):
            n_probe = self.train_dataset.n_probe
            self.train_dataset.n_probe = None
        # if one of the transforms is SampleProbe disable it
        if hasattr(self.train_dataset, "dataset") and hasattr(
            self.train_dataset.dataset, "transform"
        ):
            for transform in self.train_dataset.dataset.transform.transforms:
                if isinstance(transform, hydra.utils.get_class("boa.data.transforms.SampleProbe")):
                    n_probe = transform.n_probe
                    transform.n_probe = np.inf

        for data in self.train_dataset:
            x_sum += data.chg_labels.mean()
            x_2 += (data.chg_labels**2).mean()
            unique_atom_types.update(data.atomic_numbers.numpy().tolist())
            avg_num_neighbors += data.edge_index.shape[1] / len(data.atomic_numbers) / 2
            progress.update(1)

        x_mean = x_sum / len(self.train_dataset)
        x_var = x_2 / len(self.train_dataset) - x_mean**2
        avg_num_neighbors = int(avg_num_neighbors / len(self.train_dataset))
        # this is the avg num neighbors without the probes
        metadata = {
            "target_mean": x_mean.item(),
            "target_var": x_var.item(),
            "avg_num_neighbors": avg_num_neighbors,
            "unique_atom_types": list(unique_atom_types),
        }
        if isinstance(self.dataset, (LmdbDataset, PyscfDataset)):
            with open(self.dataset.path / "metadata_full.json", "w") as fp:
                json.dump(metadata, fp)
        else:
            with open(
                Path(self.train_dataset.dataset.root)
                / self.train_dataset.dataset.mol_name
                / "metadata_full.json",
                "w",
            ) as fp:
                json.dump(metadata, fp)
        if hasattr(self.train_dataset, "n_probe"):
            self.train_dataset.n_probe = n_probe
        if hasattr(self.train_dataset, "dataset") and hasattr(
            self.train_dataset.dataset, "transform"
        ):
            for transform in self.train_dataset.dataset.transform.transforms:
                if isinstance(transform, hydra.utils.get_class("boa.data.transforms.SampleProbe")):
                    transform.n_probe = n_probe
        return metadata

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            worker_init_fn=worker_init_fn,
            **self.dataloader_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.dataset=}, {self.num_workers=}, {self.batch_size=})"
        )


class ProbeDataModule(DataModule):
    def __init__(
        self,
        dataset: DictConfig,
        split_file: Optional[Path],
        num_workers: DictConfig,
        batch_size: DictConfig,
        n_probe: DictConfig,
        dataloader_kwargs: DictConfig,
    ):
        super().__init__(dataset, split_file, num_workers, batch_size, dataloader_kwargs)
        self.n_probe = n_probe

    def train_dataloader(self, shuffle=True):
        if self.n_probe.train > 0:
            return ProbeDataLoader(
                self.train_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                n_probe=self.n_probe.train,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )
        else:
            return DataLoader(
                self.train_dataset,
                shuffle=shuffle,
                batch_size=self.batch_size.train,
                num_workers=self.num_workers.train,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )

    def val_dataloader(self):
        if self.n_probe.val > 0:
            return ProbeDataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                n_probe=self.n_probe.val,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )
        else:
            return DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )

    def test_dataloader(self):
        if self.n_probe.test > 0:
            return ProbeDataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                n_probe=self.n_probe.test,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )
        else:
            return DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                **self.dataloader_kwargs,
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.dataset=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
            f"{self.n_probe=})"
        )
