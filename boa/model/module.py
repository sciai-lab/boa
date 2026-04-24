import logging
import time
from pathlib import Path

import numpy as np
import pyscf
import torch
from hydra.utils import instantiate
from lightning import LightningModule
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from numpy import sort
from torch_ema import ExponentialMovingAverage

from boa.model.gtos import GTOs
from scdp.common.utils import scatter
from scdp.model.basis_set import basis_from_pyscf
from scdp.model.utils import get_nmape

element_symbols_to_numbers = pyscf.data.elements.ELEMENTS_PROTON
pylogger = logging.getLogger(__name__)


def get_probe_chunks(n_probes, max_n_probe_per_pass):
    batch_size = len(n_probes)
    n_per_pass = []
    probes_to_process = []
    n_probes = torch.clone(n_probes)
    probe_indices = torch.arange(n_probes.sum(), device=n_probes.device)

    n_pass = ((n_probes).sum() / max_n_probe_per_pass).ceil().long()
    pass_start_index = 0
    pass_end_index = 0
    for _ in range(n_pass):
        current_pass = torch.zeros(batch_size, dtype=torch.long, device=n_probes.device)
        current_load = 0

        for i in range(batch_size):
            if n_probes[i] == 0:
                continue
            max_points_for_job = max_n_probe_per_pass - current_load
            points_to_process = torch.min(torch.tensor([n_probes[i], max_points_for_job]))

            current_load += points_to_process
            current_pass[i] = points_to_process
            pass_end_index += int(points_to_process)

            # in-place modification done last
            n_probes[i] -= points_to_process

            if current_load >= max_n_probe_per_pass:
                break
        n_per_pass.append(current_pass)
        probes_to_process.append(probe_indices[pass_start_index:pass_end_index])
        pass_start_index = pass_end_index
        pass_end_index = pass_start_index

    return n_pass, n_per_pass, probes_to_process


class ChgLightningModule(LightningModule):
    """
    Charge density prediction with the probe point method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.basis_info = instantiate(self.hparams.basis_info)
        self.abs_scale = self.hparams.abs_scale
        self.use_abs = self.hparams.use_abs
        if hasattr(self.hparams, "linear_basis"):
            self.linear_basis = self.hparams.linear_basis
        else:
            self.linear_basis = False
        self.construct_orbitals()
        self.model = instantiate(
            self.hparams.net,
        )
        # print(self.model)
        print(f"self.linear_basis: {self.linear_basis}")

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.hparams.train.ema.decay)
        self.distributed = (self.hparams.train.trainer.strategy == "ddp") and (
            self.hparams.train.trainer.devices > 1
        )
        self.register_buffer(
            "scale",
            torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt()
            * self.hparams.scale_factor,
        )

    def construct_orbitals(self):
        # construct GTOs
        unique_atom_types = sort(self.hparams.metadata["unique_atom_types"])
        print(f"Unique atom types: {unique_atom_types}")

        basis_dict_sym = self.basis_info.basis_dict
        basis_dict = {element_symbols_to_numbers[sym]: v for sym, v in basis_dict_sym.items()}
        basis_set = {
            anumber: basis_from_pyscf(basis_dict[anumber]) for anumber in unique_atom_types
        }

        gto_dict = {}
        for elem in unique_atom_types:
            # atomic number 0 for virtual nodes.
            gto_dict[str(elem)] = GTOs(
                **basis_set[elem],
                cutoff=self.hparams.orb_cutoff,
                use_radial_correction=self.hparams.train.model.use_radial_correction,
            )

        self.register_buffer("unique_atom_types", torch.tensor(unique_atom_types))
        self.register_buffer(
            "n_Ls", torch.tensor([len(gto_dict[str(i)].Ls) for i in unique_atom_types])
        )
        self.register_buffer(
            "n_orbitals", torch.tensor([gto_dict[str(i)].outdim for i in unique_atom_types])
        )
        self.gto_dict = torch.nn.ModuleDict(gto_dict)

        orb_index = torch.zeros(
            max(unique_atom_types) + 1, max(self.basis_info.basis_dim_per_atom), dtype=torch.bool
        )
        for i in range(len(unique_atom_types)):
            orb_index[int(unique_atom_types[i]), : self.basis_info.basis_dim_per_atom[i]] = True
        self.register_buffer("orb_index", orb_index)
        self.pbc = self.hparams.pbc

    def orbital_inference(self, batch, coeffs, n_probe, probe_coords, edge_index):
        """
        Compute chg values at given probe points using <coeffs>.
        Inputs:
            - batch: batch (bsz B) object, N atoms
            - coeffs: orbital coefficients (N, max_orbital_outdim)
            - n_probes: number of probes for each batch (B,)
            - probe_coords: probe coordinates (M, 3)
        Outputs:
            - orbitals: chg values at probe points, (M,)
        """
        unique_atom_types = torch.unique(batch.atomic_numbers)

        batch_perm = torch.argsort(batch.batch[edge_index[0]])
        edge_index = edge_index[:, batch_perm]
        coeffs = coeffs[batch_perm]

        coeffs_a, coeffs_b = coeffs.chunk(2, dim=-1)

        # sort coeffs_b like coeffs_a
        cat_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        _, inverse = torch.unique(
            cat_index, dim=1, return_inverse=True
        )  # Find unique columns and inverse indices
        perm_indices_a = inverse[: edge_index.size(1)]  # Extract indices for edge_index, Shape: M
        perm_indices_a = perm_indices_a.to(
            device=edge_index.device, dtype=torch.long
        )  # Match device and dtype
        perm_indices_b = inverse[edge_index.size(1) :]
        perm_indices_b = perm_indices_b.to(device=edge_index.device, dtype=torch.long)
        inverse_perm_a = torch.empty_like(perm_indices_a)
        inverse_perm_b = torch.empty_like(perm_indices_b)
        inverse_perm_a[perm_indices_a] = torch.arange(
            len(perm_indices_a), device=edge_index.device, dtype=torch.long
        )
        inverse_perm_b[perm_indices_b] = torch.arange(
            len(perm_indices_b), device=edge_index.device, dtype=torch.long
        )
        coeffs_b = coeffs_b[inverse_perm_b][perm_indices_a]

        coeffs = torch.cat([coeffs_a, coeffs_b], dim=-1)

        edge_preds = []
        index_probes = []
        index_edges = []
        for i in unique_atom_types:
            n_edge = scatter(
                torch.ones_like(edge_index[0]), batch.batch[edge_index[0]], dim_size=len(batch)
            )
            n_edge_i = scatter(
                (batch.atomic_numbers[edge_index[0]] == i).long(),
                torch.arange(len(batch), device=batch.atomic_numbers.device).repeat_interleave(
                    n_edge
                ),
                len(batch),
            )
            orb_index = self.orb_index[i.item()]

            edge_pred, index_probe, index_edge, _ = self.gto_dict[str(i.item())](
                probe_coords=probe_coords,
                atom_coords=batch.pos[edge_index[0, batch.atomic_numbers[edge_index[0]] == i]],
                n_probes=n_probe,
                n_atoms=n_edge_i,
                coeffs=coeffs[batch.atomic_numbers[edge_index[0]] == i][:, orb_index],
                expo_scaling=None,
                pbc=self.pbc,
                cell=batch.cell,
                return_full=True,
                second_cutoff_atom_coords=batch.pos[
                    edge_index[1, batch.atomic_numbers[edge_index[0]] == i]
                ],
            )

            current_indices = torch.where(batch.atomic_numbers[edge_index[0]] == i)[0]
            global_index_edge = current_indices[index_edge]
            edge_preds.append(edge_pred)
            index_edges.append(global_index_edge)
            index_probes.append(index_probe)

        edge_preds_a, edge_preds_b = torch.cat(edge_preds, dim=0).chunk(2, dim=-1)
        index_edges = torch.cat(index_edges, dim=0)
        index_probes = torch.cat(index_probes, dim=0)

        full_index_a = torch.cat([index_probes[None, :], edge_index[:, index_edges]], dim=0)
        full_index_b = torch.cat(
            [
                index_probes[None, :],
                edge_index[:, inverse_perm_a][:, perm_indices_b][:, index_edges],
            ],
            dim=0,
        )

        # Optimized permutation computation
        cat_index = torch.cat([full_index_a, full_index_b], dim=1)  # Shape: 3 × (2M)
        _, inverse = torch.unique(
            cat_index, dim=1, return_inverse=True
        )  # Find unique columns and inverse indices
        perm_indices_a = inverse[
            : full_index_a.size(1)
        ]  # Extract indices for full_index_b, Shape: M
        perm_indices_a = perm_indices_a.to(
            device=edge_index.device, dtype=torch.long
        )  # Match device and dtype
        perm_indices_b = inverse[full_index_a.size(1) :]
        perm_indices_b = perm_indices_b.to(device=edge_index.device, dtype=torch.long)
        inverse_perm_a = torch.empty_like(perm_indices_a)
        inverse_perm_b = torch.empty_like(perm_indices_b)
        inverse_perm_a[perm_indices_a] = torch.arange(
            len(perm_indices_a), device=edge_index.device, dtype=torch.long
        )
        inverse_perm_b[perm_indices_b] = torch.arange(
            len(perm_indices_b), device=edge_index.device, dtype=torch.long
        )

        # use a smooth L1loss instead of abs
        if self.use_abs:
            edge_preds_a = (
                torch.nn.functional.smooth_l1_loss(
                    edge_preds_a * self.abs_scale, torch.zeros_like(edge_preds_a), reduction="none"
                )
                / self.abs_scale
            )

        pred = scatter(
            ((edge_preds_a[inverse_perm_a]) * (edge_preds_b[inverse_perm_b])).sum(dim=-1),
            index_probes[inverse_perm_a],
            n_probe.sum(),
        )
        pred = pred * self.scale

        return pred

    def orbital_inference_linear(self, batch, coeffs, n_probe, probe_coords, edge_index):
        """
        Compute chg values at given probe points using <coeffs>.
        Inputs:
            - batch: batch (bsz B) object, N atoms
            - coeffs: orbital coefficients (N, max_orbital_outdim)
            - n_probes: number of probes for each batch (B,)
            - probe_coords: probe coordinates (M, 3)
        Outputs:
            - orbitals: chg values at probe points, (M,)
        """
        unique_atom_types = torch.unique(batch.atomic_numbers)

        batch_perm = torch.argsort(batch.batch[edge_index[0]])
        edge_index = edge_index[:, batch_perm]
        coeffs = coeffs[batch_perm]

        full_pred = None
        for i in unique_atom_types:
            n_edge = scatter(
                torch.ones_like(edge_index[0]), batch.batch[edge_index[0]], dim_size=len(batch)
            )
            n_edge_i = scatter(
                (batch.atomic_numbers[edge_index[0]] == i).long(),
                torch.arange(len(batch), device=batch.atomic_numbers.device).repeat_interleave(
                    n_edge
                ),
                len(batch),
            )
            orb_index = self.orb_index[i.item()]

            pred = self.gto_dict[str(i.item())](
                probe_coords=probe_coords,
                atom_coords=batch.pos[edge_index[0, batch.atomic_numbers[edge_index[0]] == i]],
                n_probes=n_probe,
                n_atoms=n_edge_i,
                coeffs=coeffs[batch.atomic_numbers[edge_index[0]] == i][:, orb_index],
                expo_scaling=None,
                pbc=self.pbc,
                cell=batch.cell,
                return_full=False,
                second_cutoff_atom_coords=None,
            )

            if full_pred is None:
                full_pred = pred
            else:
                full_pred = full_pred + pred

        pred = full_pred.sum(-1)
        pred = pred * self.scale

        return pred

    def orbital_inference_linear_fast(self, batch, coeffs, n_probe, probe_coords, edge_index):
        """
        Compute chg values at given probe points using <coeffs>.
        Inputs:
            - batch: batch (bsz B) object, N atoms
            - coeffs: orbital coefficients (N, max_orbital_outdim)
            - n_probes: number of probes for each batch (B,)
            - probe_coords: probe coordinates (M, 3)
        Outputs:
            - orbitals: chg values at probe points, (M,)
        """
        unique_atom_types = torch.unique(batch.atomic_numbers)

        # sum over edges
        coeffs = scatter(coeffs, edge_index[0], len(batch.atomic_numbers))

        print(coeffs.shape, batch.atomic_numbers.shape)

        full_pred = None
        for i in unique_atom_types:
            n_atom_i = scatter(
                (batch.atomic_numbers == i).long(),
                torch.arange(len(batch), device=batch.atomic_numbers.device).repeat_interleave(
                    batch.n_atom
                ),
                len(batch),
            )
            orb_index = self.orb_index[i.item()]

            pred = self.gto_dict[str(i.item())](
                probe_coords=probe_coords,
                atom_coords=batch.pos[batch.atomic_numbers == i],
                n_probes=n_probe,
                n_atoms=n_atom_i,
                coeffs=coeffs[batch.atomic_numbers == i][:, orb_index],
                expo_scaling=None,
                pbc=self.pbc,
                cell=batch.cell,
                return_full=False,
                second_cutoff_atom_coords=None,
            )

            if full_pred is None:
                full_pred = pred
            else:
                full_pred = full_pred + pred

        pred = full_pred
        pred = pred * self.scale

        return pred

    def forward(self, batch):
        # move everything to dtype that is not long
        coeffs, edge_index = self.model(batch)
        if self.linear_basis:
            coeffs = torch.chunk(coeffs, 2, dim=-1)[0]
            pred = self.orbital_inference_linear(
                batch, coeffs, batch.n_probe, batch.probe_coords, edge_index=edge_index
            )
        else:
            pred = self.orbital_inference(
                batch, coeffs, batch.n_probe, batch.probe_coords, edge_index=edge_index
            )

        target = batch.chg_labels
        if self.hparams.criterion == "mse":
            loss = (pred / self.scale - target / self.scale).pow(2).mean()
        else:
            loss = (pred / self.scale - target / self.scale).abs().mean()

        return loss, pred, batch.chg_labels, coeffs

    def training_step(self, batch, batch_idx):
        loss, pred, target, coeffs = self(batch)
        self.log_dict(
            {
                "loss/train": loss,
            },
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed,
        )

        if batch_idx % self.hparams.train.log_train_nmape_interval == 0:
            nmape = get_nmape(
                pred,
                target,
                torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe),
            ).mean()
            self.log_dict(
                {"nmape/train": nmape},
                batch_size=batch["cell"].shape[0],
                sync_dist=self.distributed,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, _ = self(batch)
        nmape = get_nmape(
            pred,
            target,
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe),
        ).mean()
        self.log_dict(
            {"loss/val": loss, "nmape/val": nmape},
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed,
        )
        return loss

    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        if hasattr(self, "max_n_probe_per_pass"):
            max_n_probe_per_pass = self.max_n_probe_per_pass
        else:
            max_n_probe_per_pass = 10000
        coeffs, edge_index = self.model(batch)
        n_pass, n_per_pass, probes_to_process = get_probe_chunks(
            batch.n_probe, max_n_probe_per_pass
        )
        all_preds = []
        for i_pass in range(n_pass):
            n_probe = n_per_pass[i_pass]
            probe_idx = probes_to_process[i_pass]
            probe_coords = batch.probe_coords[probe_idx]
            if self.linear_basis:
                coeffs_lin = torch.chunk(coeffs, 2, dim=-1)[0]
                pred = self.orbital_inference_linear(
                    batch, coeffs_lin, n_probe, probe_coords, edge_index=edge_index
                )
            else:
                pred = self.orbital_inference(batch, coeffs, n_probe, probe_coords, edge_index)
            all_preds.append(pred)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        self.test_durations.append(duration)
        self._set_eval_progress_bar_postfix(stage="test", duration=np.mean(self.test_durations))
        all_preds = torch.cat(all_preds, dim=0)

        if hasattr(self, "save_test_prediction_folder") and self.save_test_prediction_folder:
            assert batch.num_graphs == 1, "Saving predictions for batch_size>1 not supported."
            save_folder = Path(self.save_test_prediction_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f"{batch.id[0]}.npy"
            np.save(save_path, all_preds.cpu().numpy())

        nmape = get_nmape(
            all_preds,
            batch.chg_labels,
            torch.arange(len(batch), device=all_preds.device).repeat_interleave(batch.n_probe),
        )
        nmape_mean = nmape.mean()
        nmape = nmape.tolist()
        self.test_nmapes.extend(nmape)
        self.log_dict(
            {"nmape/test": nmape_mean},
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed,
        )
        for i, nmape in enumerate(nmape):
            if i > 0:
                pylogger.warning("Timing not supported for batch_size>1")
            atomic_numbers = batch.atomic_numbers[batch.batch == i]
            num_atoms = len(atomic_numbers)
            num_electrons = atomic_numbers.sum().item()
            self.test_results.append(
                {
                    "dataset_idx": i,
                    "id": batch.id[i],
                    "nmape": nmape,
                    "num_atoms": num_atoms,
                    "num_electrons": num_electrons,
                    "time": duration,
                }
            )
        return nmape_mean

    def on_test_epoch_end(self):
        avg_duration = sum(self.test_durations) / len(self.test_durations)
        std_duration = np.std(self.test_durations)
        self.log("test_time_avg_s", avg_duration, prog_bar=True, add_dataloader_idx=False)
        self.log("test_time_std_s", std_duration, add_dataloader_idx=False)
        print(
            f"Test duration over {len(self.test_durations)} samples: {avg_duration} ± {std_duration} seconds."
        )
        self._set_eval_progress_bar_postfix(stage="test", duration=avg_duration)

    def _set_eval_progress_bar_postfix(self, stage: str, duration: float) -> None:
        """Best-effort helper to push timing info into Lightning's tqdm bars."""
        trainer = getattr(self, "trainer", None)
        if trainer is None or not getattr(trainer, "is_global_zero", True):
            return

        callbacks = getattr(trainer, "callbacks", None)
        if not callbacks:
            return

        target_attr = {
            "val": "_val_progress_bar",
            "test": "_test_progress_bar",
        }.get(stage)
        if target_attr is None:
            return

        display_value = f"{duration:.2f}s"
        for callback in callbacks:
            if not isinstance(callback, TQDMProgressBar):
                continue
            progress_bar = getattr(callback, target_attr, None)
            if progress_bar is None or getattr(progress_bar, "disable", False):
                continue
            progress_bar.set_postfix({f"{stage}_time": display_value}, refresh=False)
            break

    def on_test_start(self):
        """Initialize list to collect NMAPE values at the start of testing."""
        self.test_nmapes = []
        self.test_durations = []
        self.test_results = []

    def on_test_end(self):
        """Save collected NMAPE values at the end of testing."""
        save_dir = Path(self.trainer.logger.log_dir)
        # Save as .pt file
        torch.save(self.test_nmapes, save_dir / "nmape_test.pt")
        # Save as .txt file
        with open(save_dir / "nmape_test.txt", "w") as f:
            for item in self.test_nmapes:
                f.write(f"{item}\n")

        with open(save_dir / "results.csv", "w") as f:
            f.write("dataset_idx,id,nmape,num_atoms,num_electrons,time\n")
            for result in self.test_results:
                f.write(
                    f"{result['dataset_idx']},{result['id']},{result['nmape']},{result['num_atoms']},{result['num_electrons']},{result['time']}\n"
                )

        # Log summary statistics
        mean_nmape = np.mean(self.test_nmapes)
        std_nmape = np.std(self.test_nmapes)
        pylogger.info(f"Test NMAPE: {mean_nmape:.4f} ± {std_nmape:.4f}")
        pylogger.info(f"Saved {len(self.test_nmapes)} NMAPE values to {save_dir}")

    def configure_optimizers(self):
        opt = instantiate(
            self.hparams.train.optim,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=opt)

        if "lr_schedule_freq" in self.hparams.train:
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.hparams.train.lr_schedule_freq,
                "monitor": self.hparams.train.monitor.metric,
            }

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.train.monitor.metric,
        }

    def on_fit_start(self):
        self.ema.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        with self.ema.average_parameters():
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        try:
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
        except Exception as e:
            print(e)
            print("Failed to load EMA state dict. Please make sure this was intended.")

    def on_validation_epoch_start(self):
        self.ema.store()
        self.ema.copy_to(self.parameters())

    def on_validation_epoch_end(self):
        self.ema.restore()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step(self.trainer.callback_metrics[self.hparams.monitor.metric])

    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.parameters())

    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"), norm_type=2.0)
        self.log("trainer/grad_norm", total_norm)
