import numpy as np
import torch
from e3nn.o3 import Irreps, Linear
from pyscf import gto
from torch import Tensor, nn
from torch_geometric.index import index2ptr
from torch_geometric.utils import to_dense_batch
import pyscf

numbers_to_element_symbols = pyscf.data.elements.ELEMENTS

from boa.data.of_data import get_coulomb_matrix, get_overlap_matrix


class MessagePassAtom(nn.Module):
    def __init__(self, inv_overlap_matrix):
        super().__init__()
        self.register_buffer("inv_overlap_matrix", inv_overlap_matrix)

    def forward(self, atom_y):
        atom_y = torch.einsum("ij, ...jc -> ...ic", self.inv_overlap_matrix, atom_y)
        return atom_y


class MessagePass(nn.Module):
    def __init__(self, inv_overlap_matrices, basis_info):
        super().__init__()
        self.inv_overlap_matrices = inv_overlap_matrices
        self.basis_info = basis_info
        self.message_pass_atoms = {}
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            self.message_pass_atoms[atom_type] = MessagePassAtom(
                self.inv_overlap_matrices[atom_type]
            )
            self.register_module(
                "message_pass_atom_" + str(atom_type), self.message_pass_atoms[atom_type]
            )

    def forward(self, x, message_passing_matrix, coeffs_batch):
        y, mask = to_dense_batch(x, coeffs_batch)
        y = torch.einsum("bij, bjc -> bic", message_passing_matrix, y)
        yy = y[mask]

        return yy
    

class MessagePassAttention(nn.Module):
    def __init__(self, inv_overlap_matrices, basis_info, channels=1):
        super().__init__()
        self.inv_overlap_matrices = inv_overlap_matrices
        self.basis_info = basis_info
        self.channels = channels

        self.message_pass_atoms = {}
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            self.message_pass_atoms[atom_type] = MessagePassAtom(
                self.inv_overlap_matrices[atom_type]
            )
            self.register_module(
                "message_pass_atom_" + str(atom_type), self.message_pass_atoms[atom_type]
            )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.channels**2, self.channels**2),
            nn.SiLU(),
            nn.Linear(self.channels**2, self.channels**2),
        )

    def forward(self, query, key, value, edge_matrices, edge_index, coeff_ind_to_node_ind):
        query, _ = to_dense_batch(query, coeff_ind_to_node_ind, max_num_nodes=edge_matrices.shape[-1])
        key, _ = to_dense_batch(key, coeff_ind_to_node_ind, max_num_nodes=edge_matrices.shape[-1])
        value, mask = to_dense_batch(value, coeff_ind_to_node_ind, max_num_nodes=edge_matrices.shape[-1])

        edge_query = query[edge_index[0]]
        edge_key = key[edge_index[1]]
        edge_value = value[edge_index[1]]

        alpha = torch.einsum(
            "eij, eic, ejm -> ecm", edge_matrices, edge_query, edge_key
        )

        alpha = self.alpha_mlp(alpha.reshape(alpha.shape[0], -1)).view(*alpha.shape)
        edge_value = torch.einsum(
            "eij, ejc -> eic", edge_matrices, edge_value
        )

        edge_value = torch.einsum(
            "ecm, eic -> eim", alpha, edge_value
        )

        # scatter the values based on edge_index
        message = torch.zeros_like(query)
        message = message.index_add(
            0, edge_index[0], edge_value
        )

        message = message[mask]
        return message


class SquaredNonlinearityAtom(nn.Module):
    def __init__(self, overlap_tensor, channel_mixing=False, n_channels=None):
        super().__init__()
        self.channel_mixing = channel_mixing
        if self.channel_mixing:
            self.channel_weights = nn.Parameter(
                torch.randn(n_channels, n_channels, n_channels) / np.sqrt(n_channels),
                requires_grad=True,
            )
        self.register_buffer("overlap_tensor", overlap_tensor)

    def forward(self, x):
        if self.channel_mixing:
            x = torch.einsum(
                "ijk, ...km, ...jn, mnc -> ...ic", self.overlap_tensor, x, x, self.channel_weights
            )
        else:
            x = torch.einsum("ijk, ...kc, ...jc -> ...ic", self.overlap_tensor, x, x)
        return x


class SquaredNonlinearity(nn.Module):
    def __init__(self, overlap_tensors, basis_info, channel_mixing=False, n_channels=None):
        super().__init__()
        self.overlap_tensors = overlap_tensors
        self.basis_info = basis_info
        self.squared_nonlinearities = {}
        for atom_type in basis_info.atomic_numbers:
            self.squared_nonlinearities[atom_type] = SquaredNonlinearityAtom(
                self.overlap_tensors[atom_type], channel_mixing, n_channels
            )
            self.register_module(
                "squared_nonlinearity_" + str(atom_type), self.squared_nonlinearities[atom_type]
            )

    def forward(self, x, atomic_number_masks):
        for i, atom_type in enumerate(self.basis_info.atomic_numbers):
            y = x[..., atomic_number_masks[atom_type], :]
            atom_y = y.view(-1, self.basis_info.basis_dim_per_atom[i], y.shape[-1])
            if not (atom_y.shape[0] == 0):
                atom_y = self.squared_nonlinearities[atom_type](atom_y)
                atom_y = atom_y.reshape(*y.shape)
                x[..., atomic_number_masks[atom_type], :] = atom_y
        return x


class StableLinearNodeOperator(nn.Module):
    def __init__(
        self, basis_info, n_channels=1, add_funcs=0, additional_irreps=Irreps([(0, (0, 1))])
    ):
        super().__init__()
        self.basis_info = basis_info
        self.stable_linear_node_operators = {}
        self.irrep_out_dims = np.zeros(max(basis_info.atomic_numbers) + 1)
        self.add_funcs = add_funcs
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            if add_funcs > 0:
                irreps_in = basis_info.irreps_per_atom[i]
                irreps_out = irreps_in + additional_irreps[i]
                irreps_out = irreps_out.sort().irreps.simplify()
                self.irrep_out_dims[atom_type] = irreps_out.dim
            else:
                irreps_in = basis_info.irreps_per_atom[i]
                irreps_out = basis_info.irreps_per_atom[i]
            self.stable_linear_node_operators[atom_type] = StableLinearNodeOperatorAtom(
                atom_type, irreps_in, irreps_out, n_channels
            )
            self.register_module(
                "stable_linear_node_operator_" + str(atom_type),
                self.stable_linear_node_operators[atom_type],
            )

    def forward(self, x, atomic_number_masks, atomic_numbers=None):
        if self.add_funcs > 0:
            dim = self.irrep_out_dims[atomic_numbers.cpu().numpy()]
            big_atomic_number_masks = {}
            coeff_to_atomic_number_mask = torch.repeat_interleave(
                atomic_numbers,
                torch.tensor(dim, device=atomic_numbers.device, dtype=torch.long),
                dim=0,
            )
            for i, atomic_number in enumerate(self.basis_info.atomic_numbers):
                coeff_mask = coeff_to_atomic_number_mask == atomic_number
                big_atomic_number_masks[atomic_number] = torch.where(coeff_mask)[0]
            new_x = torch.empty((int(dim.sum()), x.shape[-1]), device=x.device)
        else:
            new_x = None

        for i, atom_type in enumerate(self.stable_linear_node_operators.keys()):
            y = x[..., atomic_number_masks[atom_type], :]
            atom_y = y.view(-1, self.basis_info.basis_dim_per_atom[i], y.shape[-1])
            if not (atom_y.shape[0] == 0):
                atom_y = self.stable_linear_node_operators[atom_type](atom_y)
                atom_y = atom_y.reshape(-1, y.shape[-1])
                if new_x is not None:
                    new_x[..., big_atomic_number_masks[atom_type], :] = atom_y
                else:
                    x[..., atomic_number_masks[atom_type], :] = atom_y
        if new_x is not None:
            return new_x, dim, big_atomic_number_masks
        return x


class StableLinearNodeOperatorAtom(nn.Module):
    def __init__(self, atomic_number, irreps_in, irreps_out, n_channels=1):
        super().__init__()
        self.atomic_number = atomic_number

        self.linear1 = Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            internal_weights=True,
            biases=True,
        )
        self.linear1.weight.data.mul_(1 / np.sqrt(2))
        self.channel_weights1 = nn.Parameter(
            torch.randn(n_channels, n_channels) / np.sqrt(n_channels), requires_grad=True
        )

    def forward(self, x) -> Tensor:
        x = self.linear1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x @ self.channel_weights1

        return x


class L2FunctionNorm(nn.Module):
    def __init__(self, basis_info, channels):
        super().__init__()
        self.basis_info = basis_info
        self.l2_norm_atom = {}
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            self.l2_norm_atom[atom_type] = L2FunctionNormAtom(atom_type, basis_info, channels)
            self.register_module(
                "stable_linear_node_operator_" + str(atom_type), self.l2_norm_atom[atom_type]
            )

    def forward(self, x, atomic_number_masks):
        for i, atom_type in enumerate(self.l2_norm_atom.keys()):
            y = x[..., atomic_number_masks[atom_type], :]
            atom_y = y.view(-1, self.basis_info.basis_dim_per_atom[i], y.shape[-1])
            if not (atom_y.shape[0] == 0):
                atom_y = self.l2_norm_atom[atom_type](atom_y)
                atom_y = atom_y.reshape(*y.shape)
                x[..., atomic_number_masks[atom_type], :] = atom_y
        return x


class L2FunctionNormAtom(nn.Module):
    def __init__(self, atomic_number, basis_info, channels):
        super().__init__()
        aux_mol = gto.Mole()
        aux_mol.atom = [["He", [0.0, 0.0, 0.0]]]
        aux_mol.basis = basis_info.basis_dict[numbers_to_element_symbols[atomic_number]]
        aux_mol.build()

        overlap_matrix = torch.tensor(get_overlap_matrix(aux_mol), dtype=torch.float32)
        self.register_buffer("overlap_matrix", overlap_matrix)

        self.l_norm = torch.nn.Parameter(torch.ones(channels) * atomic_number, requires_grad=False)

    def forward(self, x):
        # x = x - x.mean(dim=-1, keepdim=True)
        norm = torch.einsum("ij, ...ic, ...jc -> ...c", self.overlap_matrix, x, x)
        norm = torch.sqrt(norm)
        norm = norm.unsqueeze(1)
        x = x / (norm + 1e-6)
        x = x  # * self.l_norm[None, None, :]
        return x


class L2Nonlinearity(nn.Module):
    def __init__(self, basis_info, channels):
        super().__init__()
        self.basis_info = basis_info
        self.l2_nonlinearity_atom = {}
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            self.l2_nonlinearity_atom[atom_type] = L2NonlinearityAtom(
                atom_type, basis_info, channels
            )
            self.register_module(
                "stable_linear_node_operator_" + str(atom_type),
                self.l2_nonlinearity_atom[atom_type],
            )

    def forward(self, x, atomic_number_masks):
        for i, atom_type in enumerate(self.l2_nonlinearity_atom.keys()):
            y = x[..., atomic_number_masks[atom_type], :]
            atom_y = y.view(-1, self.basis_info.basis_dim_per_atom[i], y.shape[-1])
            if not (atom_y.shape[0] == 0):
                atom_y = self.l2_nonlinearity_atom[atom_type](atom_y)
                atom_y = atom_y.reshape(*y.shape)
                x[..., atomic_number_masks[atom_type], :] = atom_y
        return x


class L2NonlinearityAtom(nn.Module):
    def __init__(self, atomic_number, basis_info, channels):
        super().__init__()
        aux_mol = gto.Mole()
        aux_mol.atom = [["He", [0.0, 0.0, 0.0]]]
        aux_mol.basis = basis_info.basis_dict[numbers_to_element_symbols[atomic_number]]
        aux_mol.build()

        self.mlp = nn.Sequential(
            nn.Linear(channels**2, channels**2),
            nn.SiLU(),
            nn.LayerNorm(channels**2),
            nn.Linear(channels**2, channels**2),
        )

        # initialize weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.mul_(1 / np.sqrt(2))
                layer.bias.data.mul_(0.0)

        overlap_matrix = torch.tensor(get_coulomb_matrix(aux_mol), dtype=torch.float32)
        self.register_buffer("overlap_matrix", overlap_matrix)

    def forward(self, x):
        l2 = torch.einsum("ij, ...ic, ...jm -> ...cm", self.overlap_matrix, x, x)
        l2 = self.mlp(l2.reshape(l2.shape[0], -1)).view(*l2.shape)
        x = x @ l2
        return x


class MatrixNormalization(nn.Module):
    """Normalizes the input so that the matrix has unit Frobenius norm."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x):
        norm = torch.norm(x, p="fro", dim=(-2, -1), keepdim=True)
        x = x / (norm + 1e-6)
        x = x * self.scale
        return x


class HeterogeneousEdgeUpdate(nn.Module):
    def __init__(self, channels, basis_info):
        super().__init__()
        self.channels = channels
        self.basis_info = basis_info

        self.heterogeneous_edge_update_0_atom = {}
        for i, atomic_number in enumerate(basis_info.atomic_numbers):
            self.heterogeneous_edge_update_0_atom[atomic_number] = HeterogeneousEdgeUpdateAtom(
                atomic_number, channels, basis_info
            )
            self.register_module(
                "heterogeneous_edge_update_0_atom_" + str(atomic_number),
                self.heterogeneous_edge_update_0_atom[atomic_number],
            )

    def forward(self, x, edge_index, edge_features_a, edge_features_b, edge_matrices, atomic_numbers):
        """
        x: Node features of shape (num_nodes, channels)
        edge_index: Edge indices of shape (2, num_edges)
        edge_features_a: Edge features for type A of shape (num_edges, channels)
        edge_features_b: Edge features for type B of shape (num_edges, channels)
        edge_matrices: Edge matrices of shape (num_edges, channels, channels)
        """
        x_edge_a = torch.zeros_like(edge_features_a)
        x_edge_b = torch.zeros_like(edge_features_b)

        for i, atomic_number in enumerate(self.basis_info.atomic_numbers):
            atomic_number_mask = atomic_numbers[edge_index[0]] == atomic_number
            if atomic_number_mask.sum() > 0:
                edge_index_masked = edge_index[:, atomic_number_mask]
                edge_features_a_masked = edge_features_a[atomic_number_mask]
                edge_features_b_masked = edge_features_b[atomic_number_mask]
                edge_matrices_masked = edge_matrices[atomic_number_mask]

                x_edge_a_masked, x_edge_b_masked = self.heterogeneous_edge_update_0_atom[atomic_number](
                    x, edge_index_masked, edge_features_a_masked, edge_features_b_masked, edge_matrices_masked
                )

                x_edge_a[atomic_number_mask] = x_edge_a_masked
                x_edge_b[atomic_number_mask] = x_edge_b_masked
        return x_edge_a, x_edge_b
            

class HeterogeneousEdgeUpdateAtom(nn.Module):
    def __init__(self, atomic_number, channels, basis_info):
        super().__init__()
        self.atomic_number = atomic_number
        self.channels = channels
        self.basis_info = basis_info

        self.channel_mlp = nn.Sequential(
            nn.Linear(2*channels**2, 2*channels**2),
            nn.SiLU(),
            nn.LayerNorm(2*channels**2),
            nn.Linear(2*channels**2, 4*channels**2),
        )
        self.norm_edge_a = MatrixNormalization()
        self.norm_edge_b = MatrixNormalization()
        self.norm_node_a = MatrixNormalization()
        self.norm_node_b = MatrixNormalization()

    def forward(self, x, edge_index, edge_features_a, edge_features_b, edge_matrices):
        x_node_a = x[edge_index[0]]
        x_node_b = x[edge_index[1]]

        edge_overlap = torch.einsum(
            "eij, eic, ejm -> ecm", edge_matrices, edge_features_a, edge_features_b
        )
        node_overlap = torch.einsum(
            "eij, eic, ejm -> ecm", edge_matrices, x_node_a, x_node_b
        )
        edge_w_a, edge_w_b, node_w_a, node_w_b = torch.chunk(
            self.channel_mlp(torch.cat([edge_overlap.reshape(edge_overlap.shape[0], -1), node_overlap.reshape(node_overlap.shape[0], -1)], dim=-1)),
            4,
            dim=-1,
        )
        edge_w_a = edge_w_a.view(*edge_overlap.shape)
        edge_w_b = edge_w_b.view(*edge_overlap.shape)
        node_w_a = node_w_a.view(*node_overlap.shape)
        node_w_b = node_w_b.view(*node_overlap.shape)

        edge_w_a = self.norm_edge_a(edge_w_a)
        edge_w_b = self.norm_edge_b(edge_w_b)
        node_w_a = self.norm_node_a(node_w_a)
        node_w_b = self.norm_node_b(node_w_b)

        x_edge_a = (edge_features_a @ edge_w_a + x_node_a @ node_w_a) / 2
        x_edge_b = (edge_features_b @ edge_w_b + x_node_b @ node_w_b) / 2

        return x_edge_a, x_edge_b

class EdgeUpdate(nn.Module):
    def __init__(self, channels, basis_info):
        super().__init__()
        self.channel_mlp = nn.Sequential(
            nn.Linear(2*channels**2, 2*channels**2),
            nn.SiLU(),
            nn.LayerNorm(2*channels**2),
            nn.Linear(2*channels**2, 4*channels**2 + 4*channels),
        )
        self.channels = channels
        self.norm_edge_a = MatrixNormalization()
        self.norm_edge_b = MatrixNormalization()
        self.norm_node_a = MatrixNormalization()
        self.norm_node_b = MatrixNormalization()


    def forward(self, x, edge_index, edge_features_a, edge_features_b, edge_matrices):
        x_node_a = x[edge_index[0]]
        x_node_b = x[edge_index[1]]

        edge_overlap = torch.einsum(
            "eij, eic, ejm -> ecm", edge_matrices, edge_features_a, edge_features_b
        )
        node_overlap = torch.einsum(
            "eij, eic, ejm -> ecm", edge_matrices, x_node_a, x_node_b
        )
        mlp_res = self.channel_mlp(torch.cat([edge_overlap.reshape(edge_overlap.shape[0], -1), node_overlap.reshape(node_overlap.shape[0], -1)], dim=-1))
        weights = mlp_res[:, :-4 * self.channels]
        factors = mlp_res[:, -4 * self.channels:]
        edge_w_a, edge_w_b, node_w_a, node_w_b = torch.chunk(
            weights,
            4,
            dim=-1,
        )
        edge_w_a = edge_w_a.view(*edge_overlap.shape)
        edge_w_b = edge_w_b.view(*edge_overlap.shape)
        node_w_a = node_w_a.view(*node_overlap.shape)
        node_w_b = node_w_b.view(*node_overlap.shape)

        edge_w_a = self.norm_edge_a(edge_w_a)
        edge_w_b = self.norm_edge_b(edge_w_b)
        node_w_a = self.norm_node_a(node_w_a)
        node_w_b = self.norm_node_b(node_w_b)

        factor_edge_a = factors[:, -4 * self.channels: -3 * self.channels][:, None, :]
        factor_edge_b = factors[:, -3 * self.channels: -2 * self.channels][:, None, :]
        factor_node_a = factors[:, -2 * self.channels: -1 * self.channels][:, None, :]
        factor_node_b = factors[:, -1 * self.channels:][:, None, :]

        x_edge_a = (factor_edge_a * (edge_features_a @ edge_w_a) + factor_node_a * (x_node_a @ node_w_a)) / 2
        x_edge_b = (factor_edge_b * (edge_features_b @ edge_w_b) + factor_node_b * (x_node_b @ node_w_b)) / 2

        return x_edge_a, x_edge_b


def to_atom_repr(x, basis_info, atomic_numbers, coeff_ind_to_node_ind):
    y, mask = to_dense_batch(x, coeff_ind_to_node_ind)
    # argsort atomic numbers
    inds = torch.argsort(atomic_numbers)
    y = y[inds]
    # find slices for each ato_atom_reprtomic number
    atom_repr = {}
    type_ptr = index2ptr(
        torch.tensor(
            basis_info.atomic_number_to_atom_index, dtype=torch.long, device=atomic_numbers.device
        )[atomic_numbers[inds]]
    )
    for i, (start, end) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
        if not (start == end):
            atom_type = int(atomic_numbers[inds[start]])
            atom_repr[atom_type] = y[
                start:end,
                : basis_info.basis_dim_per_atom[basis_info.atomic_number_to_atom_index[atom_type]],
                :,
            ]

    return atom_repr, type_ptr, inds, mask, y


def from_atom_repr(y, basis_info, atomic_numbers, type_ptr, inds, mask, atom_repr):
    y = torch.empty_like(y)
    for i, (start, end) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
        if not (start == end):
            atom_type = int(atomic_numbers[inds[start]])
            y[
                start:end,
                : basis_info.basis_dim_per_atom[basis_info.atomic_number_to_atom_index[atom_type]],
                :,
            ] = atom_repr[atom_type]

    y[inds] = y.clone()
    x = y[mask]

    return x


class BoaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        basis_info,
        inv_overlap_matrices,
        use_squared_nonlinearity=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_info = basis_info
        self.inv_overlap_matrices = inv_overlap_matrices

        self.linear0 = StableLinearNodeOperator(basis_info, in_channels)
        self.linear1 = StableLinearNodeOperator(basis_info, in_channels)
        self.linear2 = StableLinearNodeOperator(basis_info, in_channels)

        self.norm1 = L2FunctionNorm(basis_info, in_channels)

        self.l2_nonlinearity = L2Nonlinearity(basis_info, in_channels)
        self.use_squared_nonlinearity = use_squared_nonlinearity
        self.message_pass = MessagePassAttention(inv_overlap_matrices, basis_info, in_channels)
        self.edge_update = EdgeUpdate(in_channels, basis_info)

        self.integrals = {}
        for i, atom_type in enumerate(basis_info.atomic_numbers):
            self.integrals[atom_type] = torch.tensor(basis_info.integrals[i], dtype=torch.float32)
            self.register_buffer("integrals_" + str(atom_type), self.integrals[atom_type])

    def forward(
        self,
        atom_repr,
        coeff_ind_to_node_ind,
        atomic_numbers,
        type_ptr,
        inds,
        mask,
        y,
        edge_index=None,
        message_edge_index=None,
        edge_features_a=None,
        edge_features_b=None,
        edge_matrices=None,
        message_edge_matrices=None,
    ):
        atom_repr_res = {}
        for i, atom_type in enumerate(self.basis_info.atomic_numbers):
            if atom_type in atom_repr.keys():
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} before linear0"
                    )
                atom_repr_res[atom_type] = self.linear0.stable_linear_node_operators[atom_type](
                    atom_repr[atom_type].clone()
                )
                if torch.isnan(atom_repr_res[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr_res for atom type {atom_type} after linear0"
                    )
                if hasattr(self, "use_squared_nonlinearity") and self.use_squared_nonlinearity:
                    atom_repr[atom_type] = self.squared_nonlinearity.squared_nonlinearities[
                        atom_type
                    ](atom_repr[atom_type])
                else:
                    atom_repr[atom_type] = self.l2_nonlinearity.l2_nonlinearity_atom[atom_type](
                        atom_repr[atom_type]
                    )
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after nonlinearity"
                    )
                atom_repr[atom_type] = self.norm1.l2_norm_atom[atom_type](atom_repr[atom_type])
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after normalization"
                    )
                atom_repr[atom_type] = self.linear1.stable_linear_node_operators[atom_type](
                    atom_repr[atom_type]
                )
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after linear1"
                    )

        x = from_atom_repr(y, self.basis_info, atomic_numbers, type_ptr, inds, mask, atom_repr)
        x = self.message_pass(x, x, x, message_edge_matrices, message_edge_index, coeff_ind_to_node_ind)
        atom_repr, type_ptr, inds, mask, y = to_atom_repr(
            x, self.basis_info, atomic_numbers, coeff_ind_to_node_ind
        )

        for i, atom_type in enumerate(self.basis_info.atomic_numbers):
            if atom_type in atom_repr.keys():
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after message passing"
                    )
                atom_repr[atom_type] = self.message_pass.message_pass_atoms[atom_type](
                    atom_repr[atom_type]
                )
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after message passing atom"
                    )
                atom_repr[atom_type] = (
                    self.linear2.stable_linear_node_operators[atom_type](atom_repr[atom_type])
                    + atom_repr_res[atom_type]
                )
                if torch.isnan(atom_repr[atom_type]).any():
                    raise ValueError(
                        f"NaN detected in atom_repr for atom type {atom_type} after linear2"
                    )

        x = from_atom_repr(
            y, self.basis_info, atomic_numbers, type_ptr, inds, mask, atom_repr
        )
        x = to_dense_batch(x, coeff_ind_to_node_ind, max_num_nodes=max(self.basis_info.basis_dim_per_atom))[0]
        edge_features_a, edge_features_b = self.edge_update(
            x, edge_index, edge_features_a, edge_features_b, edge_matrices
        )
        if torch.isnan(edge_features_a).any() or torch.isnan(edge_features_b).any():
            raise ValueError("NaN detected in edge features after edge update")
        return atom_repr, edge_features_a, edge_features_b
