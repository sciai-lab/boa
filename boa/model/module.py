import math
import numpy as np
import pyscf
import torch
from lightning import LightningModule
from hydra.utils import instantiate
from boa.data.basis_info import BasisInfo
from boa.data.of_data import OFBatch, OFData, ToTorch
from torch_ema import ExponentialMovingAverage
from mldft.utils.molecules import build_molecule_np
from scdp.common.utils import scatter
from scdp.model.utils import get_nmape
from scdp.model.basis_set import basis_from_pyscf, get_basis_set, transform_basis_set, aug_etb_for_basis

from boa.model.gtos import GTOs
from boa.model.net.boa_net import BOA
from boa.model.gtos import MAX_L

element_symbols_to_numbers = pyscf.data.elements.ELEMENTS_PROTON


def move_batch_to_dtype(batch, dtype):
    """
    Move all tensors in the batch to the specified dtype, except for long and bool tensors.
    """
    for k in batch.keys:
        v = batch[k]
        if hasattr(v, "dtype") and v.dtype != torch.long and v.dtype != torch.bool:
            batch[k] = v.to(dtype)
    if hasattr(batch, "of_batch"):
        of_batch = batch.of_batch
        for k in of_batch.keys():
            v = of_batch[k]
            if hasattr(v, "dtype") and v.dtype != torch.long and v.dtype != torch.int and v.dtype != torch.bool:
                of_batch[k] = v.to(dtype=dtype, device=batch.coords.device)
            elif hasattr(v, "dtype"):
                of_batch[k] = v.to(device=batch.coords.device)

    return batch

class ChgLightningModule(LightningModule):
    """
    Charge density prediction with the probe point method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if not "eSCN" in self.hparams.net._target_:
            self.basis_info = instantiate(self.hparams.basis_info)
        self.construct_orbitals()
        num_neighbors = self.hparams.metadata["avg_num_neighbors"]
        if "eSCN" in self.hparams.net._target_:
            self.model = instantiate(
                self.hparams.net, 
                num_neighbors=num_neighbors, 
                expo_trainable=self.hparams.expo_trainable,
                max_n_Ls=self.max_n_Ls,
                max_n_orbitals_per_L=self.max_n_orbitals_per_L
            )
        else:
            self.model = instantiate(
                self.hparams.net, 
            )
        print(self.model)
        
        self.ema = ExponentialMovingAverage(
            self.parameters(), decay=self.hparams.ema.decay
        )        
        self.distributed = ((self.hparams.trainer.strategy == "ddp") and 
                            (self.hparams.trainer.devices > 1))
        self.register_buffer("scale", torch.FloatTensor([self.hparams.metadata["target_var"]]).sqrt())

    def construct_orbitals(self):
        # construct GTOs
        unique_atom_types = self.hparams.metadata['unique_atom_types']

        if self.hparams.dft_wt_aug:
            if "eSCN" in self.hparams.net._target_:
                basis_set = transform_basis_set(get_basis_set(self.hparams.dft_basis_set))
                basis_set = aug_etb_for_basis(
                    basis_set, 
                    beta=self.hparams.beta, 
                    lmax_restriction=self.hparams.lmax_restriction,
                    lmax_relax=self.hparams.lmax_relax if 'lmax_relax' in self.hparams else 0,
                )
            else:
                basis_dict_sym = self.basis_info.basis_dict
                basis_dict = {element_symbols_to_numbers[sym]: v for sym, v in basis_dict_sym.items()}
                basis_set = {anumber: basis_from_pyscf(basis_dict[anumber]) for anumber in unique_atom_types}
        else:
            if not ("eSCN" in self.hparams.net._target_):
                basis_dict_sym = self.basis_info.basis_dict
                basis_dict = {element_symbols_to_numbers[sym]: v for sym, v in basis_dict_sym.items()}
                basis_set = {anumber: basis_from_pyscf(basis_dict[anumber]) for anumber in unique_atom_types}
            else:
                basis_set = transform_basis_set(get_basis_set(self.hparams.basis_set))

        # if self.hparams.add_basis_functions is not None:
        #     bfuncs = self.hparams.add_basis_functions
        #     for elem in unique_atom_types:
        #         bset = basis_set[elem]
        #         Ls = bfuncs[elem]["Ls"]
        #         expos = bfuncs[elem]["expos"]
        #         coeffs = bfuncs[elem]["coeffs"]
                
        #         bset['Ls'].extend(Ls)
        #         bset['expos'].extend(expos)
        #         bset['coeffs'].extend(coeffs)
        #         bset['contraction'].extend(np.arange(max(bset['contraction']) + 1, max(bset['contraction']) + 1 + len(Ls)).tolist())

        print(basis_set)

        initial_guess_basis_set = self.hparams.get('initial_guess_basis_set', None)
        print(f"Using initial guess basis set: {initial_guess_basis_set}")
        if initial_guess_basis_set is not None:
            initial_guess_basis_set = transform_basis_set(
                get_basis_set(initial_guess_basis_set)
            )
            basis_dict_sym = BasisInfo.from_atomic_numbers_with_even_tempered_basis([1, 6, 7, 8, 9], self.hparams.initial_guess_basis_set, beta=1.5).basis_dict
            basis_dict = {element_symbols_to_numbers[sym]: v for sym, v in basis_dict_sym.items()}
            initial_guess_basis_set = {anumber: basis_from_pyscf(basis_dict[anumber]) for anumber in unique_atom_types}
            for anumber in unique_atom_types:
                # remove contraction from initial guess basis set
                initial_guess_basis_set[anumber]['contraction'] = None

        vbasis = basis_set[self.hparams.vnode_elem]
                    
        if self.hparams.uncontracted:
            for v in basis_set.values():
                v['contraction'] = None
            vbasis['contraction'] = None
            
        gto_dict = {}
        for elem in unique_atom_types:
            # atomic number 0 for virtual nodes.
            if elem == 0:
                gto_dict['0'] = GTOs(**vbasis, cutoff=self.hparams.orb_cutoff)
            else:
                gto_dict[str(elem)] = GTOs(**basis_set[elem], cutoff=self.hparams.orb_cutoff)

        if initial_guess_basis_set is not None:
            initial_guess_gto_dict = {}
            for elem in unique_atom_types:
                initial_guess_gto_dict[str(elem)] = GTOs(
                    **initial_guess_basis_set[elem], cutoff=self.hparams.orb_cutoff,
                )
        
        self.register_buffer('unique_atom_types', torch.tensor(unique_atom_types))
        self.register_buffer('n_Ls', torch.tensor([len(gto_dict[str(i)].Ls) for i in unique_atom_types]))
        self.register_buffer('n_orbitals', torch.tensor([gto_dict[str(i)].outdim for i in unique_atom_types]))
        self.gto_dict = torch.nn.ModuleDict(gto_dict)

        if initial_guess_basis_set is not None:
            self.initial_guess_gto_dict = torch.nn.ModuleDict(initial_guess_gto_dict)
            self.initial_guess_basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis([1, 6, 7, 8, 9], self.hparams.initial_guess_basis_set, beta=1.5)
            print(self.hparams.dataset_statistics)
            self.sad_guesser = SADGuesser.from_dataset_statistics(
                                                instantiate(self.hparams.dataset_statistics),
                                                self.initial_guess_basis_info,
                                            )
        
        if not self.hparams.uncontracted:
            contract_dict = {}
            for k, v in gto_dict.items():
                gto = v
                con_per_l = []
                for l in range(MAX_L + 1):
                    l_mask = gto.Ls == l
                    num_contracted = (scatter(l_mask.to(dtype=torch.int64), torch.tensor(gto.contraction, dtype=torch.int64), int(gto.contraction.max().item())+1)).sum().item()
                    num_contracted = num_contracted - torch.unique(gto.contraction[l_mask]).numel() 
                    con_per_l.append(num_contracted)
                contract_dict[k] = torch.tensor(con_per_l, dtype=torch.int64)
        else:
            contract_dict = {str(k): [0] * (self.Lmax + 1) for k in unique_atom_types}

        self.Lmax = max([gto.Lmax for gto in self.gto_dict.values()])
        self.max_n_Ls = max([len(gto.Ls) - contract_dict[k].sum() for k, gto in self.gto_dict.items()])
        self.max_n_orbitals_per_L = torch.stack(
            [x.n_orbitals_per_L - contract_dict[k] for k, x in self.gto_dict.items()]).max(dim=0)[0]
        self.outdim_per_L = torch.stack(
            [x.n_orbitals_per_L - contract_dict[k] for k, x in self.gto_dict.items()]) * (2 * torch.arange(MAX_L+1) + 1)[None, :]
        self.max_outdim_per_L = self.max_n_orbitals_per_L * (2 * torch.arange(len(self.max_n_orbitals_per_L)) + 1)
        self.max_outdim = int(self.max_outdim_per_L.sum())

        if "eSCN" in self.hparams.net._target_:      
            orb_index = torch.zeros(max(unique_atom_types)+1, self.max_outdim, dtype=torch.bool)
            offsets = torch.cat([torch.tensor([0]), torch.cumsum(self.max_outdim_per_L, dim=0)])        
            L_index = torch.zeros(max(unique_atom_types)+1, self.max_n_Ls, dtype=torch.bool)
            for i, (k, v) in enumerate(gto_dict.items()):
                index = torch.cat(
                    [torch.arange(offsets[l], offsets[l]+self.outdim_per_L[i][l]) for l in range(self.Lmax+1)])
                orb_index[int(k), index] = True
                L_index[int(k), :len(v.Ls) - contract_dict[k].sum()] = True
            self.register_buffer('orb_index', orb_index)
            self.register_buffer('L_index', L_index)
        else:
            orb_index = torch.zeros(max(unique_atom_types)+1, self.max_outdim, dtype=torch.bool)
            for i in range(len(unique_atom_types)):
                orb_index[int(unique_atom_types[i]), :self.basis_info.basis_dim_per_atom[i]] = True
            self.register_buffer('orb_index', orb_index)
        self.pbc = self.hparams.pbc
    
    def predict_coeffs(self, batch):
        """
        predict coefficient for GTO basis functions.
        """
        if self.model.quadratic_readout_v2:
            if isinstance(self.model, BOA):
                coeffs, expo_scaling, edge_index = self.model(batch.of_batch)
            else:
                coeffs, expo_scaling, edge_index = self.model(batch)
        else:
            if isinstance(self.model, BOA):
                coeffs = self.model(batch.of_batch)
                expo_scaling = None
            else:
                coeffs, expo_scaling = self.model(batch)
            edge_index = batch.edge_index
        
        n_orbs = self.n_orbitals[
            (batch.atom_types.repeat(len(self.unique_atom_types), 1).T == 
             self.unique_atom_types).nonzero()[:, 1]]
        batch_n_orbs = scatter(n_orbs, batch.batch, len(batch))

        if coeffs.dim() == 3:
            if self.model.quadratic_readout_v2:
                batch_n_edge = scatter(torch.ones_like(edge_index[0]), batch.batch[edge_index[0]], dim_size=len(batch))
                normalization = batch_n_orbs.repeat_interleave(
                    batch_n_edge).sqrt().view(-1, 1)
            else:
                normalization = batch_n_orbs.repeat_interleave(
                    batch.n_atom + batch.n_vnode).sqrt().view(-1, 1)
            inds = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
            # inds = [0,1,2,3,5,6,7,8]
            coeffs[..., inds]  = coeffs[..., inds] / normalization.unsqueeze(-1)
            # coeffs = coeffs / normalization.unsqueeze(-1)
        else:
            normalization = batch_n_orbs.repeat_interleave(
                batch.n_atom + batch.n_vnode).sqrt().view(-1, 1)
            coeffs = coeffs / normalization 

        if expo_scaling is not None:
            # range from 0.5 to 2.0
            expo_scaling = 1.5 / (1 + torch.exp(-expo_scaling + math.log(2))) + 0.5
        
        return coeffs, expo_scaling, edge_index
    
    def orbital_inference(self, batch, coeffs, expo_scaling, n_probe, probe_coords, edge_index):
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
        unique_atom_types = torch.unique(batch.atom_types)
        
        if coeffs.dim() == 3:
            if self.model.quadratic_readout_v2:
                batch_perm = torch.argsort(batch.batch[edge_index[0]])
                edge_index = edge_index[:, batch_perm]
                coeffs = coeffs[batch_perm]

                coeffs_a, coeffs_b = coeffs.chunk(2, dim=-1)

                # sort coeffs_b like coeffs_a
                cat_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  
                _, inverse = torch.unique(cat_index, dim=1, return_inverse=True)  # Find unique columns and inverse indices
                perm_indices_a = inverse[:edge_index.size(1)]  # Extract indices for edge_index, Shape: M
                perm_indices_a = perm_indices_a.to(device=edge_index.device, dtype=torch.long)  # Match device and dtype
                perm_indices_b = inverse[edge_index.size(1):]
                perm_indices_b = perm_indices_b.to(device=edge_index.device, dtype=torch.long)
                inverse_perm_a = torch.empty_like(perm_indices_a)
                inverse_perm_b = torch.empty_like(perm_indices_b)
                inverse_perm_a[perm_indices_a] = torch.arange(len(perm_indices_a), device=edge_index.device, dtype=torch.long)
                inverse_perm_b[perm_indices_b] = torch.arange(len(perm_indices_b), device=edge_index.device, dtype=torch.long)
                coeffs_b = coeffs_b[inverse_perm_b][perm_indices_a]

                coeffs = torch.cat([coeffs_a, coeffs_b], dim=-1)

                edge_preds = []
                # edge_preds_b = []
                index_probes = []
                # index_probes_b = []
                index_edges = []
                # index_edges_b = []
                masks = []
                for i in unique_atom_types:
                    n_edge = scatter(torch.ones_like(edge_index[0]), batch.batch[edge_index[0]], dim_size=len(batch))
                    n_edge_i = scatter(
                        (batch.atom_types[edge_index[0]] == i).long(),
                        torch.arange(len(batch), device=batch.atom_types.device).repeat_interleave(
                        n_edge), len(batch)
                    )
                    orb_index = self.orb_index[i.item()]
                    if expo_scaling is not None:
                        L_index = self.L_index[i.item()]

                    
                    edge_pred, index_probe, index_edge, mask = self.gto_dict[str(i.item())](
                        probe_coords=probe_coords, 
                        atom_coords=batch.coords[edge_index[0, batch.atom_types[edge_index[0]] == i]],
                        n_probes=n_probe,
                        n_atoms=n_edge_i,
                        coeffs=coeffs[batch.atom_types[edge_index[0]] == i][:, orb_index],
                        expo_scaling=None,
                        pbc=self.pbc, cell=batch.cell, return_full=True,
                        second_cutoff_atom_coords=batch.coords[edge_index[1, batch.atom_types[edge_index[0]] == i]],
                    )
                    # edge_pred_b, index_probe_b, index_edge_b, mask_b = self.gto_dict[str(i.item())](
                    #     probe_coords=probe_coords, 
                    #     atom_coords=batch.coords[edge_index[1, batch.atom_types[edge_index[1]] == i]],
                    #     n_probes=n_probe,
                    #     n_atoms=n_edge_i,
                    #     coeffs=coeffs_b[batch.atom_types[edge_index[1]] == i][:, orb_index],
                    #     expo_scaling=None,
                    #     pbc=self.pbc, cell=batch.cell, return_full=True,
                    #     second_cutoff_atom_coords=batch.coords[edge_index[0, batch.atom_types[edge_index[1]] == i]],
                    # )

                    current_indices = torch.where(batch.atom_types[edge_index[0]] == i)[0]
                    # current_indices_b = torch.where(batch.atom_types[edge_index[1]] == i)[0]
                    global_index_edge = current_indices[index_edge]
                    # global_index_edge_b = current_indices_b[index_edge_b]
                    edge_preds.append(edge_pred)
                    # edge_preds_b.append(edge_pred_b)
                    index_edges.append(global_index_edge)
                    # index_edges_b.append(global_index_edge_b)
                    index_probes.append(index_probe)
                    # index_probes_b.append(index_probe_b)
                    # index_edges.append(torch.arange(batch.edge_index.shape[1], device=batch.edge_index.device, dtype=torch.long)[batch.atom_types[batch.edge_index[0]] == i][index_edge])
                    # masks.append(mask)
                edge_preds_a, edge_preds_b = torch.cat(edge_preds, dim=0).chunk(2, dim=-1)
                # edge_preds_b = torch.cat(edge_preds_b, dim=0)
                index_edges = torch.cat(index_edges, dim=0)
                # index_edges_b = torch.cat(index_edges_b, dim=0)
                index_probes = torch.cat(index_probes, dim=0)
                # index_probes_b = torch.cat(index_probes_b, dim=0)

                # # # edge_index = batch.edge_index
                # # # full_index_a = torch.cat([index_probes_a[None, :], edge_index[:, index_edges_a]], dim=0)
                # # # full_index_b = torch.cat([index_probes_b[None, :], edge_index[:, index_edges_b].flip(0)], dim=0)
                # # # # find permutation indices to reverse the edge direction
                # # # full_index_a_dict = {tuple(edge): idx for idx, edge in enumerate(full_index_a.t().tolist())}
                # # # perm_indices = torch.tensor([full_index_a_dict[tuple(edge)] for edge in full_index_b.t().tolist()], 
                # # #                             device=edge_index.device, dtype=torch.long)

                # Existing tensor constructions (unchanged)
                #edge_index = batch.edge_index
                full_index_a = torch.cat([index_probes[None, :], edge_index[:, index_edges]], dim=0)
                full_index_b = torch.cat([index_probes[None, :], edge_index[:, inverse_perm_a][:, perm_indices_b][:, index_edges]], dim=0)

                # # print number of indices in full_index_a that are not in full_index_b
                # j = 0
                # for i in range(full_index_a.size(1)):
                #     if not torch.any(torch.all(full_index_a[:, i:i+1] == full_index_b, dim=0)):
                #         j += 1
                #         print(f"Index {i} in full_index_a not found in full_index_b. {full_index_a[:, i:i+1]}")
                # print("full_index_a shape:", full_index_a.shape, "full_index_b shape:", full_index_b.shape, "j:", j)

                # Optimized permutation computation
                cat_index = torch.cat([full_index_a, full_index_b], dim=1)  # Shape: 3 × (2M)
                _, inverse = torch.unique(cat_index, dim=1, return_inverse=True)  # Find unique columns and inverse indices
                perm_indices_a = inverse[:full_index_a.size(1)]  # Extract indices for full_index_b, Shape: M
                perm_indices_a = perm_indices_a.to(device=edge_index.device, dtype=torch.long)  # Match device and dtype
                perm_indices_b = inverse[full_index_a.size(1):]
                perm_indices_b = perm_indices_b.to(device=edge_index.device, dtype=torch.long)  
                inverse_perm_a = torch.empty_like(perm_indices_a)
                inverse_perm_b = torch.empty_like(perm_indices_b)
                inverse_perm_a[perm_indices_a] = torch.arange(len(perm_indices_a), device=edge_index.device, dtype=torch.long)
                inverse_perm_b[perm_indices_b] = torch.arange(len(perm_indices_b), device=edge_index.device, dtype=torch.long)
                # print("unique indices shape:", _.shape)
                # print("inverse shape:", inverse.shape)
                # print("full_index_a:", full_index_a[:, perm_indices_b][:, :3])
                # print("full_index_b:", full_index_b[:, perm_indices_a][:, :3])
                # print("test a", torch.all(full_index_a == _[:, perm_indices_a]))
                # print("test b", torch.all(full_index_b == _[:, perm_indices_b]))
                # assert torch.all(index_probes[inverse_perm_a] == index_probes[inverse_perm_b]), \
                #     "Index probes for atom and edge probes do not match after permutation."
                # assert torch.all(edge_index[:, index_edges][:, inverse_perm_a] ==
                #        edge_index[:, index_edges][:, inverse_perm_b]), \
                #     "Edge indices for atom and edge probes do not match after permutation."
                # index_edges = torch.cat(index_edges, dim=0)edge_pred_a
                # masks = torch.cat(masks, dim=0)

                # # n_edge = scatter(torch.ones_like(batch.edge_index[0]), batch.batch[batch.edge_index[0]], dim_size=len(batch))
                # # device = probe_coords.device
                # # n_pairs = n_probe * n_edge
                # # n_total_pairs = n_pairs.sum()

                # # index_offset_p = torch.cumsum(n_probe, dim=0) - n_probe
                # # index_offset_p = torch.repeat_interleave(index_offset_p, n_pairs)
                # # index_offset_a = torch.cumsum(n_edge, dim=0) - n_edge
                # # index_offset_a = torch.repeat_interleave(index_offset_a, n_pairs)
                # # index_offset_pair = torch.cumsum(n_pairs, dim=0) - n_pairs
                # # index_offset_pair = torch.repeat_interleave(index_offset_pair, n_pairs)
                # # pair_count = torch.arange(n_total_pairs, device=device) - index_offset_pair
                # # n_edge_expand = torch.repeat_interleave(n_edge, n_pairs)

                # # index_probe = torch.div(pair_count, n_edge_expand, rounding_mode='trunc').long() + index_offset_p
                # # index_atom = (pair_count % n_edge_expand).long() + index_offset_a

                # # # alternative way of building index_atom


                # # edge_index = batch.edge_index
                # # full_index = torch.cat([index_probes[None, :], edge_index[:, index_edges]], dim=0)
                # # full_index_reverse = torch.cat([index_probes[None, :], edge_index[:, index_edges].flip(0)], dim=0)
                # # # find permutation indices to reverse the edge direction
                # # full_index_dict = {tuple(edge): idx for idx, edge in enumerate(full_index.t().tolist())}
                # # perm_indices = torch.tensor([full_index_dict[tuple(edge)] for edge in full_index_reverse.t().tolist()], 
                # #                             device=edge_index.device, dtype=torch.long)

                # edge_index = batch.edge_index
                # e = edge_index.size(1)
                # edge_list = edge_index.t().tolist()
                # edge_dict = {tuple(edge): idx for idx, edge in enumerate(edge_list)}
                # perm_indices = [edge_dict[(edge_index[1, i].item(), edge_index[0, i].item())] 
                #                 for i in range(e)]
                # perm = torch.tensor(perm_indices, device=edge_index.device, dtype=torch.long)
                # perm = perm[index_edges]

                # # print shapes
                # print("edge_preds shape:", edge_preds.shape)
                # print("index_probes shape:", index_probes.shape)
                # print("index_edges shape:", index_edges.shape)
                # print("edge_index shape:", batch.edge_index.shape)
                # print("masks shape:", masks.shape)

                # edge_preds_a = torch.abs(edge_preds_a)
                # use a smooth L1loss instead of abs
                # edge_preds_a = torch.nn.functional.smooth_l1_loss(edge_preds_a*1e3, torch.zeros_like(edge_preds_a), reduction='none')/1e3

                # pred = (edge_preds*edge_preds[perm]).sum(dim=-1)
                pred = scatter(((edge_preds_a[inverse_perm_a])*(edge_preds_b[inverse_perm_b])).sum(dim=-1), index_probes[inverse_perm_a], n_probe.sum())
                # Detect nans
                if torch.isnan(pred).any():
                    print("NaN detected in prediction.")
                    print("Number of NaNs:", torch.isnan(pred).sum().item())
                    print("Nan probe coordinates:", probe_coords[torch.isnan(pred)])
                    print("Number of nan edge features:", torch.isnan(edge_preds_a).sum().item())
                    print("Number of nan edge features b:", torch.isnan(edge_preds_b).sum().item())
                    print("Number of nan coeffs:", torch.isnan(coeffs).sum().item())
                # edge_pred = torch.zeros(probe_coords.shape[0], batch.edge_index.shape[1], coeffs.shape[-1], device=coeffs.device, dtype=coeffs.dtype)
                # for i in unique_atom_types:
                #     n_atom_i = scatter(
                #         (batch.atom_types == i).long(), 
                #         torch.arange(len(batch), device=batch.atom_types.device).repeat_interleave(
                #         batch.n_atom + batch.n_vnode), len(batch)
                #     )
                #     orb_index = self.orb_index[i.item()]
                #     if expo_scaling is not None:
                #         L_index = self.L_index[i.item()]
                #     gtos, index_atom, index_probe = self.gto_dict[str(i.item())](
                #         probe_coords=probe_coords, 
                #         atom_coords=batch.coords[batch.atom_types == i], 
                #         n_probes=n_probe,
                #         n_atoms=n_atom_i,
                #         #coeffs=coeffs[batch.atom_types[batch.edge_index[0]] == i][:, orb_index],
                #         expo_scaling=expo_scaling[batch.atom_types == i][:, L_index] if expo_scaling is not None else None,
                #         pbc=self.pbc, cell=batch.cell
                #     )
                #     print("gtos shape:", gtos.shape)
                #     vals = (gtos.unsqueeze(-1) * coeffs[batch.atom_types[batch.edge_index[0]] == i][:, orb_index][index_atom]).sum(dim=1)
                #     print("vals shape:", vals.shape)
                #     edge_pred[:, batch.atom_types[batch.edge_index[0]] == i, :] = scatter(
                #                                                     vals, 
                #                                                     index_probe, n_probe.sum()
                #                                                 )
                #     #edge_pred[:, batch.atom_types[batch.edge_index[0]] == i, :] = torch.einsum('pg, egc -> pec', gtos, coeffs[batch.atom_types[batch.edge_index[0]] == i][:, orb_index])
                # edge_index = batch.edge_index
                # e = edge_index.size(1)
                # edge_list = edge_index.t().tolist()
                # edge_dict = {tuple(edge): idx for idx, edge in enumerate(edge_list)}
                # perm_indices = [edge_dict[(edge_index[1, i].item(), edge_index[0, i].item())] 
                #                 for i in range(e)]
                # perm = torch.tensor(perm_indices)
                # pred = torch.einsum('pec, pec -> pc', edge_pred, edge_pred[:, perm, :]).sum(dim=-1)
                # #pred = scatter((edge_pred * edge_pred[:, perm, :]).permute(1, 0, 2), batch.batch[edge_index[0]], dim_size=len(batch)).sum(dim=-1)
                # print("pred shape:", pred.shape)
            else:
                pred = torch.zeros(probe_coords.shape[0], coeffs.shape[-1], device=coeffs.device, dtype=coeffs.dtype)
                for i in unique_atom_types:
                    n_atom_i = scatter(
                        (batch.atom_types == i).long(), 
                        torch.arange(len(batch), device=batch.atom_types.device).repeat_interleave(
                        batch.n_atom + batch.n_vnode), len(batch)
                    )
                    orb_index = self.orb_index[i.item()]
                    if expo_scaling is not None:
                        L_index = self.L_index[i.item()]
                    pred += self.gto_dict[str(i.item())](
                        probe_coords=probe_coords, 
                        atom_coords=batch.coords[batch.atom_types == i], 
                        n_probes=n_probe,
                        n_atoms=n_atom_i,
                        coeffs=coeffs[batch.atom_types == i][:, orb_index],
                        expo_scaling=expo_scaling[batch.atom_types == i][:, L_index] if expo_scaling is not None else None,
                        pbc=self.pbc, cell=batch.cell
                    )
                pred = (pred**2).sum(dim=-1)
        else:        
            pred = torch.zeros(probe_coords.shape[0], device=coeffs.device, dtype=coeffs.dtype) 
            for i in unique_atom_types:
                n_atom_i = scatter(
                    (batch.atom_types == i).long(), 
                    torch.arange(len(batch), device=batch.atom_types.device).repeat_interleave(
                    batch.n_atom + batch.n_vnode), len(batch)
                )
                orb_index = self.orb_index[i.item()]
                if expo_scaling is not None:
                    L_index = self.L_index[i.item()]
                pred += self.gto_dict[str(i.item())](
                    probe_coords=probe_coords, 
                    atom_coords=batch.coords[batch.atom_types == i], 
                    n_probes=n_probe,
                    n_atoms=n_atom_i,
                    coeffs=coeffs[batch.atom_types == i][:, orb_index],
                    expo_scaling=expo_scaling[batch.atom_types == i][:, L_index] if expo_scaling is not None else None,
                    pbc=self.pbc, cell=batch.cell
                )
        pred = pred * self.scale

        if hasattr(self, 'initial_guess_gto_dict'):
            data_list = []
            for i in range(len(batch)):
                mol = build_molecule_np(charges = batch.atom_types[batch.batch==i].cpu().numpy(),
                                        positions = batch.coords[batch.batch==i].cpu().numpy(), basis = self.initial_guess_basis_info.basis_dict)
                of_data = ToTorch(device=self.device)(OFData.minimal_sample_from_mol(mol, self.initial_guess_basis_info))
                data_list.append(of_data)
            of_data = OFBatch.from_data_list(data_list, ["coeffs", "atomic_numbers"])
            init_coeffs = self.sad_guesser(of_data).to(batch.coords.device)
            for i in unique_atom_types:
                n_atom_i = scatter(
                    (batch.atom_types == i).long(), 
                    torch.arange(len(batch), device=batch.atom_types.device).repeat_interleave(
                    batch.n_atom + batch.n_vnode), len(batch)
                )
                pred += self.initial_guess_gto_dict[str(i.item())](
                    probe_coords=probe_coords, 
                    atom_coords=of_data.pos[of_data.atomic_numbers == i],
                    n_probes=n_probe,
                    n_atoms=n_atom_i,
                    coeffs=init_coeffs[of_data.atomic_numbers[of_data.coeff_ind_to_node_ind] == i].view(-1, self.initial_guess_basis_info.basis_dim_per_atom[self.initial_guess_basis_info.atomic_number_to_atom_index[i.item()]]),
                    expo_scaling=None,
                    pbc=self.pbc, cell=batch.cell
                )
                # print(self.initial_guess_gto_dict[str(i.item())](
                #     probe_coords=probe_coords, 
                #     atom_coords=of_data.pos[of_data.atomic_numbers == i],
                #     n_probes=n_probe,
                #     n_atoms=n_atom_i,
                #     coeffs=init_coeffs[of_data.atomic_numbers[of_data.coeff_ind_to_node_ind] == i].view(-1, self.initial_guess_basis_info.basis_dim_per_atom[i.item()]),
                #     expo_scaling=None,
                #     pbc=self.pbc, cell=batch.cell
                # ).mean())
    
        return pred
    
    def forward(self, batch):
        # move everything to dtype that is not long 
        batch = move_batch_to_dtype(batch, self.dtype)
        coeffs, expo_scaling, edge_index = self.predict_coeffs(batch)
        pred = self.orbital_inference(batch, coeffs, expo_scaling, batch.n_probe, batch.probe_coords, edge_index=edge_index)        
        
        target = batch.chg_labels
        if self.hparams.criterion == 'mse':
            loss = (pred / self.scale - target / self.scale).pow(2).mean()
        else:
            loss = (pred / self.scale - target / self.scale).abs().mean()
            
        return loss, pred, batch.chg_labels, coeffs, expo_scaling
    
    def training_step(self, batch, batch_idx):
        loss, _, _, coeffs, scaling = self(batch)
        self.log_dict({
            "loss/train": loss,
            }, 
            batch_size=batch["cell"].shape[0], 
            sync_dist=self.distributed
        )   
        
        if scaling is not None:
            self.log_dict({
                "trainer/scaling_mean": scaling.mean(),
                "trainer/scaling_std": scaling.std()
                }, 
                batch_size=batch["cell"].shape[0], 
                sync_dist=self.distributed
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        nmape = get_nmape(
            pred, target, 
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe)
        ).mean()
        self.log_dict({
            "loss/val": loss,
            "nmape/val": nmape
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, target, _, _ = self(batch)
        nmape = get_nmape(
            pred, target, 
            torch.arange(len(batch), device=target.device).repeat_interleave(batch.n_probe)
        ).mean()
        self.log_dict({
            "loss/test": loss,
            "nmape/test": nmape
            }, 
            batch_size=batch["cell"].shape[0],
            sync_dist=self.distributed
            )
        return loss

    def configure_optimizers(self):
        opt = instantiate(
            self.hparams.optim,
            params=self.parameters(),
            _convert_="partial",
        )
        scheduler = instantiate(self.hparams.lr_scheduler, optimizer=opt)
        
        if 'lr_schedule_freq' in self.hparams:
            scheduler = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.hparams.lr_schedule_freq,
                'monitor': self.hparams.monitor.metric
            }
            
        return {"optimizer": opt, "lr_scheduler": scheduler, 'monitor': self.hparams.monitor.metric}
    
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
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), float('inf'), norm_type=2.0)
        self.log('trainer/grad_norm', total_norm)
        # log gradient norm for all parameters with parameter name
        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         self.log(f'grad_norm/{name}', param.grad.norm(2).item())
        # # print how many gradients are nan
        # if any(p.grad is not None and torch.isnan(p.grad).any() for p in self.parameters()):
        #     print("NaN detected in gradients.")
        #     for i, p in enumerate(self.parameters()):
        #         if p.grad is not None and torch.isnan(p.grad).any():
        #             print(f"Parameter {i} has NaN gradients.")