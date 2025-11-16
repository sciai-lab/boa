import bisect
import os
import pickle
from pathlib import Path

import ase
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from mldft.ml.data.components.of_data import OFData, Representation
from mldft.utils.molecules import build_molecule_np
from scdp.common.typing import assert_is_instance as aii
from scdp.data.data import AtomicData
from scdp.scripts.preprocess import get_atomic_number_table_from_zs


class LmdbDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.transform = transform

        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._keys = []
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries
                    # in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.env = self.connect_db(self.path)
            num_entries = aii(self.env.stat()["entries"], int)
            # If "length" encoded as ascii is present, we have one fewer
            # data than the stats suggest
            if self.env.begin().get("length".encode("ascii")) is not None:
                num_entries -= 1
            self._keys = list(range(num_entries))
            self.num_samples = num_entries

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx].begin().get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )

            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(f"{self._keys[idx]}".encode("ascii"))
            data_object = pickle.loads(datapoint_pickled)

        if self.transform:
            data_object = self.transform(data_object)

        return data_object

    def get_metadata(self, num_samples):
        pass

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


"""
Taken from https://github.com/ccr-cheng/InfGCN-pytorch/blob/main/datasets/small_density.py
and modified to fit into the SCDP framework.
"""

ATOM_TYPES = {
    "benzene": torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    "ethanol": torch.LongTensor([0, 0, 2, 1, 1, 1, 1, 1, 1]),
    "phenol": torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1]),
    "resorcinol": torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1]),
    "ethane": torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1]),
    "malonaldehyde": torch.LongTensor([2, 0, 0, 0, 2, 1, 1, 1, 1]),
}
TYPE_TO_Z = {0: 6, 1: 1, 2: 8}

BOHR_TO_ANGSTROM = 0.529177


class SmallDensityDataset(Dataset):
    def __init__(
        self,
        root,
        mol_name,
        split,
        basis_info,
        convert_to_angstrom=True,
        n_probe=None,
        transform=None,
    ):
        """
        Density dataset for small molecules in the MD datasets.
        Note that the validation and test splits are the same.
        :param root: data root
        :param mol_name: name of the molecule
        :param split: data split, can be 'train', 'test'
        """
        super(SmallDensityDataset, self).__init__()
        assert mol_name in (
            "benzene",
            "ethanol",
            "phenol",
            "resorcinol",
            "ethane",
            "malonaldehyde",
        )
        self.root = root
        self.mol_name = mol_name
        self.split = split

        self.n_grid = 50  # number of grid points along each dimension
        self.grid_size = 20.0  # box size in Bohr
        self.data_path = os.path.join(root, mol_name, f"{mol_name}_{split}")

        self.atom_type = ATOM_TYPES[mol_name]
        self.atom_charges = torch.tensor(
            [TYPE_TO_Z[i.item()] for i in self.atom_type], dtype=torch.long
        )
        self.atom_coords = np.load(os.path.join(self.data_path, "structures.npy"))
        self.densities = self._convert_fft(
            np.load(os.path.join(self.data_path, "dft_densities.npy"))
        )
        self.grid_coord = self._generate_grid()

        self.convert_to_angstrom = convert_to_angstrom
        if self.convert_to_angstrom:
            self.grid_coord = self.grid_coord * BOHR_TO_ANGSTROM
            self.grid_size = self.grid_size * BOHR_TO_ANGSTROM
            self.atom_coords = self.atom_coords * BOHR_TO_ANGSTROM
            self.densities = self.densities / (BOHR_TO_ANGSTROM**3)

        self.basis_info = basis_info
        self.n_probe = n_probe
        self.transform = transform

    def _convert_fft(self, fft_coeff):
        # The raw data are stored in Fourier basis, we need to convert them back.
        print(f"Precomputing {self.split} density from FFT coefficients ...")
        fft_coeff = torch.FloatTensor(fft_coeff).to(torch.complex64)
        d = fft_coeff.view(-1, self.n_grid, self.n_grid, self.n_grid)
        hf = self.n_grid // 2
        # first dimension
        d[:, :hf] = (d[:, :hf] - d[:, hf:] * 1j) / 2
        d[:, hf:] = torch.flip(d[:, 1 : hf + 1], [1]).conj()
        d = torch.fft.ifft(d, dim=1)
        # second dimension
        d[:, :, :hf] = (d[:, :, :hf] - d[:, :, hf:] * 1j) / 2
        d[:, :, hf:] = torch.flip(d[:, :, 1 : hf + 1], [2]).conj()
        d = torch.fft.ifft(d, dim=2)
        # third dimension
        d[..., :hf] = (d[..., :hf] - d[..., hf:] * 1j) / 2
        d[..., hf:] = torch.flip(d[..., 1 : hf + 1], [3]).conj()
        d = torch.fft.ifft(d, dim=3)
        return torch.flip(d.real.view(-1, self.n_grid**3), [-1]).detach()

    def _generate_grid(self):
        x = torch.linspace(self.grid_size / self.n_grid, self.grid_size, self.n_grid)
        return torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1).view(-1, 3).detach()

    def subsample_grid(self, of_data, n_probe):
        assert isinstance(n_probe, int), "n_probe must be an integer"
        if n_probe < of_data.n_probe:
            indices = torch.randperm(of_data.n_probe)[:n_probe]
            of_data.probe_coords = of_data.probe_coords[indices]
            of_data.chg_labels = of_data.chg_labels[indices]
            of_data.n_probe[:] = n_probe
        return of_data

    def __getitem__(self, item):
        charges = self.atom_charges
        coords = self.atom_coords[item]

        mol = build_molecule_np(
            charges=charges.numpy(),
            positions=coords.astype(np.float64),
            basis=self.basis_info.basis_dict,
        )

        of_data = OFData.minimal_sample_from_mol(mol, self.basis_info)

        of_data.add_item("n_probe", torch.tensor([self.grid_coord.shape[0]]), Representation.NONE)
        of_data.add_item(
            "cell", (torch.eye(3) * self.grid_size).view(1, 3, 3), Representation.VECTOR
        )
        of_data.add_item("probe_coords", self.grid_coord, Representation.VECTOR)
        of_data.add_item("chg_labels", self.densities[item], Representation.SCALAR)
        of_data.add_item(
            "n_atom", torch.tensor([len(self.atom_charges)], dtype=int), Representation.NONE
        )
        of_data.add_item("atom_types", self.atom_charges, Representation.NONE)

        if self.n_probe is not None:
            of_data = self.subsample_grid(of_data, self.n_probe)

        if self.transform is not None:
            of_data = self.transform(of_data)

        return of_data

    def __len__(self):
        return self.atom_coords.shape[0]


def read_xyz(path):
    with open(path, "r") as f:
        lines = f.readlines()
    elements = []
    atom_coords = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            elements.append(parts[0])
            atom_coords.append([float(x) for x in parts[1:]])
    atom_types = [ase.data.atomic_numbers[elem] for elem in elements]
    atom_types = torch.tensor(atom_types)
    atom_coords = torch.tensor(atom_coords)
    return atom_types, atom_coords


def read_pyscf(folder_path):
    density = np.load(folder_path / "rho_22.npy")
    density = density.reshape(*np.loadtxt(folder_path / "grid_sizes_22.dat", dtype=int))
    density = torch.tensor(density, dtype=torch.float32)
    # load atom_types and atom_coords from xyz file at centered.xyz
    atom_types, atom_coords = read_xyz(folder_path / "centered.xyz")
    cell = np.loadtxt(folder_path / "box.dat")
    cell = torch.tensor(cell, dtype=torch.float32)

    return atom_types, atom_coords, cell, density, None, None


class PyscfDataset(Dataset):
    def __init__(self, path, transform=None, prefix="dsgdb9nsd_"):
        super().__init__()
        self.path = Path(path)
        self.z_table = get_atomic_number_table_from_zs(np.arange(100).tolist())
        self.transform = transform
        self.prefix = prefix

    def __getitem__(self, index):
        # fill index with 0 to 6 digits
        item_path = self.path / f"{self.prefix}{index:06d}"
        data = read_pyscf(item_path)

        data = AtomicData.build_graph_with_vnodes(*data, self.z_table, vnode_method="None")
        if self.transform is not None:
            data = self.transform(data)
        return data
