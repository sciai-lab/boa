from typing import Tuple

import pyscf
import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from boa.data.basis_info import BasisInfo
from boa.model.net.boa_block_stack import BoaBlockStack

numbers_to_element_symbols = pyscf.data.elements.ELEMENTS


class NodeEmbedding(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.register_buffer(
            "basis_dim_per_atom",
            torch.tensor(basis_info.basis_dim_per_atom, dtype=torch.long),
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )
        self.is_scalar_mask = basis_info.l_per_basis_func == 0
        self.scalar_dims = []
        for index in basis_info.atom_ind_to_basis_function_ind:
            self.scalar_dims.append(self.is_scalar_mask[index].sum().item())
        self.scalar_dims = torch.tensor(self.scalar_dims, dtype=torch.long)
        self.embedding = nn.Embedding(
            len(basis_info.atomic_numbers), channels * self.scalar_dims.max().item()
        )

    def forward(self, atomic_numbers: Tensor, coeff_ind_to_node_ind: Tensor) -> Tensor:
        """
        :param atomic_numbers: Tensor of shape (batch_size, n_atoms)
        :param x: Tensor of shape (batch_size, n_channels)
        :return: Tensor of shape (batch_size, n_channels + n_scalar_features)
        """
        x = torch.zeros(
            self.basis_dim_per_atom[self.atomic_number_to_atom_index[atomic_numbers]].sum().item(),
            self.channels,
            device=atomic_numbers.device,
            dtype=self.embedding.weight.dtype,
        )
        x, mask = to_dense_batch(
            x, coeff_ind_to_node_ind, max_num_nodes=max(self.basis_dim_per_atom)
        )
        batch_size = atomic_numbers.shape[0]
        scalar_features = self.embedding(self.atomic_number_to_atom_index[atomic_numbers]).view(
            batch_size, -1, self.channels
        )
        for i, a in enumerate(self.basis_info.atomic_numbers):
            x[atomic_numbers == a, : self.scalar_dims[i]] = scalar_features[
                atomic_numbers == a, : self.scalar_dims[i]
            ]

        x = x[mask]

        return x


class ReducedEdgeEmbedding(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.node_embedding_a = NodeEmbedding(basis_info, channels=channels)
        self.node_embedding_b = NodeEmbedding(basis_info, channels=channels)

    def forward(self, batch) -> Tensor:
        """
        :param batch
        :return: Tensor of shape (n_edges, n_channels)
        """

        edge_index = batch.edge_index
        atomic_numbers = batch.atomic_numbers
        coeff_ind_to_node_ind = batch.coeff_ind_to_node_ind

        node_features_a = self.node_embedding_a(atomic_numbers, coeff_ind_to_node_ind)
        node_features_b = self.node_embedding_b(atomic_numbers, coeff_ind_to_node_ind)

        node_features_a = to_dense_batch(
            node_features_a,
            coeff_ind_to_node_ind,
            max_num_nodes=max(self.basis_info.basis_dim_per_atom),
        )[0]
        node_features_b = to_dense_batch(
            node_features_b,
            coeff_ind_to_node_ind,
            max_num_nodes=max(self.basis_info.basis_dim_per_atom),
        )[0]

        edge_features_a = torch.zeros(
            edge_index.shape[1],
            node_features_a.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=node_features_a.dtype,
        )
        edge_features_b = torch.zeros(
            edge_index.shape[1],
            node_features_b.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=node_features_b.dtype,
        )

        edge_features_a[edge_index[0] == edge_index[1]] = node_features_a
        edge_features_b[edge_index[0] == edge_index[1]] = node_features_b

        return torch.cat([edge_features_a, edge_features_b], dim=-1), edge_index


class EdgeEmbedding(nn.Module):
    """EdgeEmbedding class for embedding edge features based on atomic numbers. Each edge feature is initialized
    with a learnable embedding vector corresponding to the atomic numbers of the nodes it connects.
    """

    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.register_buffer(
            "basis_dim_per_atom",
            torch.tensor(basis_info.basis_dim_per_atom, dtype=torch.long),
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )
        self.is_scalar_mask = basis_info.l_per_basis_func == 0
        self.scalar_dims = []
        for index in basis_info.atom_ind_to_basis_function_ind:
            self.scalar_dims.append(self.is_scalar_mask[index].sum().item())
        self.scalar_dims = torch.tensor(self.scalar_dims, dtype=torch.long)

        self.embedding = nn.Embedding(
            len(basis_info.atomic_numbers) ** 2, channels * self.scalar_dims.max().item()
        )

        edge_to_embedding_index = torch.arange(
            len(basis_info.atomic_numbers) ** 2,
            dtype=torch.long,
        ).view(len(basis_info.atomic_numbers), len(basis_info.atomic_numbers))
        self.register_buffer("edge_to_embedding_index", edge_to_embedding_index)

    def forward(self, batch) -> Tensor:
        atomic_numbers = batch.atomic_numbers
        edge_index = batch.edge_index

        atomic_numbers_a = atomic_numbers[edge_index[0]]
        atomic_numbers_b = atomic_numbers[edge_index[1]]

        edge_a = torch.zeros(
            edge_index.shape[1],
            self.basis_dim_per_atom.max().item(),
            self.channels,
            device=edge_index.device,
            dtype=self.embedding.weight.dtype,
        )
        edge_b = torch.zeros(
            edge_index.shape[1],
            self.basis_dim_per_atom.max().item(),
            self.channels,
            device=edge_index.device,
            dtype=self.embedding.weight.dtype,
        )

        scalar_features_a = self.embedding(
            self.edge_to_embedding_index[
                self.atomic_number_to_atom_index[atomic_numbers_a],
                self.atomic_number_to_atom_index[atomic_numbers_b],
            ]
        ).view(-1, self.scalar_dims.max().item(), self.channels)
        scalar_features_b = self.embedding(
            self.edge_to_embedding_index[
                self.atomic_number_to_atom_index[atomic_numbers_b],
                self.atomic_number_to_atom_index[atomic_numbers_a],
            ]
        ).view(-1, self.scalar_dims.max().item(), self.channels)
        for i, a in enumerate(self.basis_info.atomic_numbers):
            for j, b in enumerate(self.basis_info.atomic_numbers):
                edge_index_mask = (atomic_numbers_a == a) & (atomic_numbers_b == b)
                edge_a[edge_index_mask, : self.scalar_dims[i]] = scalar_features_a[
                    edge_index_mask, : self.scalar_dims[i]
                ]
                edge_b[edge_index_mask, : self.scalar_dims[j]] = scalar_features_b[
                    edge_index_mask, : self.scalar_dims[j]
                ]

        return torch.cat([edge_a, edge_b], dim=-1), edge_index


class EdgeEmbeddingV2(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.edge_embedding = EdgeEmbedding(basis_info, channels=channels)
        self.node_cor = EdgeEmbedding(basis_info, channels=channels)

    def forward(self, batch) -> Tensor:
        edge_index = batch.edge_index

        edge_features = self.edge_embedding(batch)[0]
        node_cor_per_edge = self.node_cor(batch)[0]

        edge_features_a, edge_features_b = (
            edge_features[..., : self.channels],
            edge_features[..., self.channels :],
        )
        node_cor_a_per_edge, node_cor_b_per_edge = (
            node_cor_per_edge[..., : self.channels],
            node_cor_per_edge[..., self.channels :],
        )

        self_loop_mask = edge_index[0] == edge_index[1]

        node_cor_a = torch.zeros(
            self_loop_mask.sum(),
            node_cor_a_per_edge.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=edge_features_a.dtype,
        )
        node_cor_b = torch.zeros(
            self_loop_mask.sum(),
            node_cor_b_per_edge.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=edge_features_b.dtype,
        )

        node_cor_a.index_add(0, edge_index[0], node_cor_a_per_edge)
        node_cor_b.index_add(0, edge_index[0], node_cor_b_per_edge)

        out_features_a = torch.empty_like(edge_features_a)
        out_features_b = torch.empty_like(edge_features_b)

        # takes only is correct if the edge_index is sorted
        out_features_a[~self_loop_mask] = edge_features_a[~self_loop_mask]
        out_features_b[~self_loop_mask] = edge_features_b[~self_loop_mask]
        out_features_a[self_loop_mask] = edge_features_a[self_loop_mask] + node_cor_a
        out_features_b[self_loop_mask] = edge_features_b[self_loop_mask] + node_cor_b

        return torch.cat([out_features_a, out_features_b], dim=-1), edge_index


class BOA(nn.Module):
    def __init__(
        self,
        basis_info: BasisInfo,
        boa_stack: BoaBlockStack,
        initial_guess_module: nn.Module,
        direct_gs_prediction: bool = False,
        num_orbitals: int = 0,
    ) -> None:
        super().__init__()
        self.basis_info = basis_info
        self.boa_stack = boa_stack

        self.num_channels = boa_stack.blocks[0].in_channels

        self.direct_gs_prediction = direct_gs_prediction
        self.num_orbitals = num_orbitals
        self.node_embedding = NodeEmbedding(basis_info, channels=self.num_channels)
        self.edge_embedding = ReducedEdgeEmbedding(basis_info, channels=self.num_channels)

        self.initial_guess_module = initial_guess_module

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        if self.node_embedding is not None:
            x = self.node_embedding(batch.atomic_numbers, batch.coeff_ind_to_node_ind)

        edge_features = self.edge_embedding(batch)[0]
        edge_features_a, edge_features_b = (
            edge_features[..., : self.num_channels],
            edge_features[..., self.num_channels :],
        )

        init_guess_delta, edge_features_a, edge_features_b = self.boa_stack(
            x,
            batch.coeff_ind_to_node_ind,
            batch.atomic_numbers,
            batch.edge_index,
            batch.message_edge_index,
            edge_features_a=edge_features_a,
            edge_features_b=edge_features_b,
            edge_matrices=batch.edge_matrices,
            message_edge_matrices=batch.message_edge_matrices,
        )

        full_edge_index = batch.edge_index
        edge_features_a = edge_features_a.view(
            edge_features_a.shape[0], edge_features_a.shape[1], -1, self.num_orbitals
        ).mean(-2)
        edge_features_b = edge_features_b.view(
            edge_features_b.shape[0], edge_features_b.shape[1], -1, self.num_orbitals
        ).mean(-2)

        init_guess_edge = self.initial_guess_module(batch)[0]
        init_guess_edge_a, init_guess_edge_b = (
            init_guess_edge[..., 0][..., None],
            init_guess_edge[..., 1][..., None],
        )
        init_guess_delta = torch.cat(
            [edge_features_a, init_guess_edge_a, edge_features_b, init_guess_edge_b], dim=-1
        )

        return init_guess_delta, full_edge_index
