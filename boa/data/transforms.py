from functools import partial
import numpy as np
import torch
from boa.data.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from boa.data.overlap_matrix import OverlapMatrix
from mldft.ml.data.components.of_data import Representation
from mldft.ofdft.basis_integrals import get_overlap_matrix
from mldft.utils.molecules import build_molecule_np, build_molecule_ofdata
from mldft.ml.data.components.convert_transforms import AddRadiusEdgeIndex as BaseAddRadiusEdgeIndex, apply_to_attributes, dtype_map


class MasterTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_object):
        for transform in self.transforms:
            data_object = transform(data_object)
        return data_object


class SampleProbe:
    def __init__(self, n_probe: int):
        self.n_probe = n_probe

    def __call__(self, sample):
        sample = sample.sample_probe(n_probe=min(self.n_probe, sample.n_probe))
        return sample

class ConvertToOFData:
    def __init__(self, basis_info: BasisInfo):
        self.basis_info = basis_info

    def __call__(self, sample):
        mol = build_molecule_np(charges = sample.atom_types,
            positions = sample.coords, basis = self.basis_info.basis_dict)
        of_data = OFData.minimal_sample_from_mol(mol, self.basis_info)

        of_data.add_item(
            "n_probe",
            sample.n_probe,
            Representation.NONE
        )
        of_data.add_item(
            "probe_coords",
            sample.probe_coords,
            Representation.NONE
        )
        of_data.add_item(
            "chg_labels",
            sample.chg_labels,
            Representation.NONE
        )
        of_data.add_item(
            "n_atom",
            sample.n_atom,
            Representation.NONE
        )
        of_data.add_item(
            "n_vnode",
            sample.n_vnode,
            Representation.NONE
        )
        of_data.add_item(
            "cell",
            sample.cell,
            Representation.NONE
        )
        return of_data


class AddMessagePassingMatrix:
    """Adds the coulomb matrix to the sample."""

    def __init__(
        self, basis_info: BasisInfo, type: str = "overlap", remove_diagonal: bool = False
    ):
        """
        Args:
            basis_info: The basis information in the BasisInfo format.
        """
        super().__init__()
        self.basis_info = basis_info
        self.type = type
        self.remove_diagonal = remove_diagonal

    def __call__(self, sample):
        """
        Args:
            sample: the molecule in the OFData format
        """
        mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
        mol_aux = mol.copy()

        basis_dict = mol_aux.basis
        mol_aux.basis = basis_dict
        mol_aux.build()

        if self.type == "overlap":
            message_passing_matrix = torch.as_tensor(
                get_overlap_matrix(mol_aux), dtype=torch.float64
            )

        if self.remove_diagonal:
            # Remove the diagonal elements
            if isinstance(sample.n_basis_per_atom, torch.Tensor):
                n_basis_per_atom = sample.n_basis_per_atom.cpu().numpy()
            else:
                n_basis_per_atom = sample.n_basis_per_atom
            inds = np.cumsum(n_basis_per_atom)
            inds = np.insert(inds, 0, 0)
            for i in range(len(inds) - 1):
                message_passing_matrix[inds[i] : inds[i + 1], inds[i] : inds[i + 1]] = 0.0

        sample.add_item(
            "message_passing_matrix",
            OverlapMatrix(message_passing_matrix),
            Representation.BILINEAR_FORM,
        )

        return sample
    

class AddEdgeMatrices:
    def __init__(self, basis_info: BasisInfo, name: str = "edge_matrices", edge_name: str = "edge_index"):
        self.basis_info = basis_info
        self.name = name
        self.edge_name = edge_name

    def __call__(self, sample):
        assert hasattr(sample, "message_passing_matrix"), "Sample must have message_passing_matrix attribute. Apply AddMessagePassingMatrix first."

        edge_matrices = []
        for edge in sample[self.edge_name].T:
            mask_a = sample.coeff_ind_to_node_ind == edge[0].item()
            mask_b = sample.coeff_ind_to_node_ind == edge[1].item()
            edge_matrix = torch.zeros(
                (self.basis_info.basis_dim_per_atom.max(), self.basis_info.basis_dim_per_atom.max()),
                dtype=torch.float64
            )
            edge_matrix[:sample.n_basis_per_atom[edge[0].item()], :sample.n_basis_per_atom[edge[1].item()]] = (
                sample.message_passing_matrix[mask_a][:, mask_b]
            )
            edge_matrices.append(edge_matrix)
        edge_matrices = torch.stack(edge_matrices, dim=0)
        sample.add_item(
            self.name,
            edge_matrices,
            Representation.NONE,
        )
        return sample
    

class AddRadiusEdgeIndex:
    def __init__(self, radius: float, name: str = "edge_index"):
        self.radius = radius
        self.name = name
        self.base_add_radius_edge_index = BaseAddRadiusEdgeIndex(radius)

    def __call__(self, sample):
        sample = self.base_add_radius_edge_index(sample)
        edge_index = sample.edge_index
        # Remove the original edge_index
        sample.delete_item("edge_index")
        sample.add_item(
            self.name,
            edge_index,
            Representation.NONE,
        )
        return sample


def tensor_or_array_to_torch(
    x: torch.Tensor | np.ndarray,
    device=None,
    float_dtype: np.dtype | torch.dtype | str | None = None,
) -> torch.Tensor:
    dtype = dtype_map(x.dtype, float_dtype=float_dtype)
    if isinstance(x, OverlapMatrix):
        return OverlapMatrix(torch.as_tensor(x.data, dtype=dtype, device=device))
    else:
        return torch.as_tensor(x, dtype=dtype, device=device)


def to_torch(
    sample: OFData, device=None, float_dtype: np.dtype | torch.dtype | str | None = None
) -> OFData:
    """Convert all numpy arrays in the sample to torch tensors.

    Args:
        sample: The sample.
        device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
        float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
            and "torch.float32" are supported to enable hydra support.

    Returns:
        The sample with all numpy arrays converted to torch tensors.
    """
    keys = []
    for key in sample.keys():
        if isinstance(getattr(sample, key), np.ndarray):
            if not getattr(sample, key).dtype == np.object_:
                keys.append(key)
        elif isinstance(getattr(sample, key), torch.Tensor):
            keys.append(key)
    func = partial(tensor_or_array_to_torch, device=device, float_dtype=float_dtype)
    apply_to_attributes(func, sample, keys)
    return sample
    

class ToTorch:
    """Convert all numpy arrays in the sample to torch tensors."""

    def __init__(
        self, device: torch.device = None, float_dtype: np.dtype | torch.dtype | None = None
    ):
        """Initialize the transform.

        Args:
            device: The device to put the tensors on. Defaults to None, i.e. the pytorch default device.
            float_dtype: The dtype to use for float dtypes. Defaults to the pytorch default dtype. For strings, "torch.float64"
                and "torch.float32" are supported.
        """
        self.device = device
        self.float_dtype = float_dtype

    def __call__(self, sample: OFData) -> OFData:
        """Apply the transform to the sample.

        See :func:`to_torch` for details.
        """
        return to_torch(sample, device=self.device, float_dtype=self.float_dtype)