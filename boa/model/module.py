import pyscf
import torch
from hydra.utils import instantiate
from lightning import LightningModule
from torch_ema import ExponentialMovingAverage

from boa.model.gtos import MAX_L, GTOs
from scdp.common.utils import scatter
from scdp.model.basis_set import basis_from_pyscf
from scdp.model.utils import get_nmape

element_symbols_to_numbers = pyscf.data.elements.ELEMENTS_PROTON


class ChgLightningModule(LightningModule):
    """
    Charge density prediction with the probe point method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.basis_info = instantiate(self.hparams.basis_info)
        self.abs_scale = self.hparams.abs_scale
        self.construct_orbitals()
        self.model = instantiate(
            self.hparams.net,
        )
        print(self.model)

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
        unique_atom_types = self.hparams.metadata["unique_atom_types"]

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

        contract_dict = {}
        for k, v in gto_dict.items():
            gto = v
            con_per_l = []
            for l in range(MAX_L + 1):
                l_mask = gto.Ls == l
                num_contracted = (
                    (
                        scatter(
                            l_mask.to(dtype=torch.int64),
                            torch.tensor(gto.contraction, dtype=torch.int64),
                            int(gto.contraction.max().item()) + 1,
                        )
                    )
                    .sum()
                    .item()
                )
                num_contracted = num_contracted - torch.unique(gto.contraction[l_mask]).numel()
                con_per_l.append(num_contracted)
            contract_dict[k] = torch.tensor(con_per_l, dtype=torch.int64)

        self.Lmax = max([gto.Lmax for gto in self.gto_dict.values()])
        self.max_n_Ls = max(
            [len(gto.Ls) - contract_dict[k].sum() for k, gto in self.gto_dict.items()]
        )
        self.max_n_orbitals_per_L = torch.stack(
            [x.n_orbitals_per_L - contract_dict[k] for k, x in self.gto_dict.items()]
        ).max(dim=0)[0]
        self.outdim_per_L = (
            torch.stack([x.n_orbitals_per_L - contract_dict[k] for k, x in self.gto_dict.items()])
            * (2 * torch.arange(MAX_L + 1) + 1)[None, :]
        )
        self.max_outdim_per_L = self.max_n_orbitals_per_L * (
            2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1
        )
        self.max_outdim = int(self.max_outdim_per_L.sum())

        orb_index = torch.zeros(max(unique_atom_types) + 1, self.max_outdim, dtype=torch.bool)
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
        edge_preds_a = (
            torch.nn.functional.smooth_l1_loss(
                edge_preds_a * self.abs_scale, torch.zeros_like(edge_preds_a), reduction="none"
            )
            / self.abs_scale
        )

        # pred = (edge_preds*edge_preds[perm]).sum(dim=-1)
        pred = scatter(
            ((edge_preds_a[inverse_perm_a]) * (edge_preds_b[inverse_perm_b])).sum(dim=-1),
            index_probes[inverse_perm_a],
            n_probe.sum(),
        )
        pred = pred * self.scale

        return pred

    def forward(self, batch):
        # move everything to dtype that is not long
        coeffs, edge_index = self.model(batch)
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
        loss, pred, target, _ = self(batch)
        nmape = get_nmape(
            pred,
            target,
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe),
        ).mean()
        self.log_dict(
            {"loss/test": loss, "nmape/test": nmape},
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed,
        )
        return loss

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
