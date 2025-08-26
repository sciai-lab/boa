import numpy as np
import torch
from pyscf import gto
from torch import Tensor, nn
import pyscf

from boa.data.basis_info import BasisInfo
from mldft.ofdft.basis_integrals import get_overlap_matrix
from boa.model.net.boa_block import from_atom_repr, to_atom_repr

numbers_to_element_symbols = pyscf.data.elements.ELEMENTS

class BoaBlockStack(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        n_blocks: int,
        basis_info: BasisInfo,
        hidden_channels: int,
        use_squared_nonlinearity=False,
        hierarchical: bool = False
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.basis_info = basis_info
        self.hierarchical = hierarchical

        self.inv_overlap_matrices = {}
        for atomic_number in basis_info.atomic_numbers:
            element = numbers_to_element_symbols[atomic_number]
            aux_mol = gto.Mole()
            aux_mol.atom = [["He", [0.0, 0.0, 0.0]]]
            aux_mol.basis = basis_info.basis_dict[element]
            aux_mol.build()

            overlap_matrix = get_overlap_matrix(aux_mol)
            inv_overlap_matrix = torch.inverse(torch.tensor(overlap_matrix, dtype=torch.float64)) #+ 1e-4 * torch.eye(overlap_matrix.shape[0], dtype=torch.float64))
            self.inv_overlap_matrices[atomic_number] = inv_overlap_matrix.to(
                dtype=torch.float32
            )

        self.blocks = nn.ModuleList(
            [
                block(
                    hidden_channels,
                    hidden_channels,
                    basis_info,
                    self.inv_overlap_matrices,
                    use_squared_nonlinearity,
                )
                for _ in range(n_blocks - 1)
            ]
        )
        self.blocks.append(
            block(
                hidden_channels,
                hidden_channels,
                basis_info,
                self.inv_overlap_matrices,
                use_squared_nonlinearity,
            )
        )
        

    def forward(
        self, x, coeff_ind_to_node_ind, atomic_numbers, edge_index, message_edge_index, edge_features_a=None, edge_features_b=None, edge_matrices=None, message_edge_matrices=None
    ) -> Tensor:
        atom_repr, type_ptr, inds, mask, y = to_atom_repr(
            x, self.basis_info, atomic_numbers, coeff_ind_to_node_ind
        )
        if self.hierarchical:
            atom_repr_out = {atom_type: torch.zeros_like(atom_repr[atom_type]) for atom_type in atom_repr.keys()}
        for i in range(self.n_blocks):
            atom_repr, edge_features_a, edge_features_b = self.blocks[i](
                atom_repr,
                coeff_ind_to_node_ind,
                atomic_numbers,
                type_ptr,
                inds,
                mask,
                y,
                edge_index,
                message_edge_index,
                edge_features_a,
                edge_features_b,
                edge_matrices,
                message_edge_matrices,
            )
            if self.hierarchical:
                for i, atom_type in enumerate(self.basis_info.atomic_numbers):
                    if atom_type in atom_repr.keys():
                        atom_repr_out[atom_type] += atom_repr[atom_type]

        if self.hierarchical:
            atom_repr = atom_repr_out
            
        x = from_atom_repr(y, self.basis_info, atomic_numbers, type_ptr, inds, mask, atom_repr)
        
        return x, edge_features_a, edge_features_b 