"""The :class:`BasisInfo` class holds ML-relevant information about the basis."""
from typing import Dict

import numpy as np
from mldft.ml.data.components.basis_info import _irreps_list_to_array
from mldft.ofdft.basis_integrals import get_normalization_vector
import pyscf
import torch
from e3nn.o3 import Irreps
from omegaconf import OmegaConf

from mldft.utils.molecules import build_mol_with_even_tempered_basis, construct_aux_mol
from mldft.ml.data.components.basis_info import BasisInfo as BaseBasisInfo

class BasisInfo(BaseBasisInfo):
    @classmethod
    def from_atomic_numbers_with_even_tempered_basis(
        cls,
        atomic_numbers: list[int] | np.ndarray,
        basis: str | list[str] = "6-31G(2df,p)",
        beta: float = 2.5,
        even_tempered = True,
        uncontracted: bool = False,
        add_basis_functions: Dict[str, list[float]] | None = None,
    ) -> "BasisInfo":
        """Construct a BasisInfo object from a list of atomic numbers, using an even-tempered
        basis. The list or array of atomic numbers should have unique values!

        Args:
            atomic_numbers: List of atomic numbers of the atoms in the basis.
            basis: Basis set to use before converting to even-tempered basis. This matters and should be the same as was
                used during Kohn-Sham calculations. Defaults to "6-31G(2df,p)".
            beta: Exponent factor :math:`\\beta` of the even-tempered basis set. Defaults to 2.5.

        Returns:
            BasisInfo object.
        """
        if isinstance(basis, str):
            basis_list = [basis]
        else:
            basis_list = basis

        assert len(atomic_numbers) == len(
            set(atomic_numbers)
        ), "atomic_numbers values must be unique."
        atomic_numbers = np.array(atomic_numbers, dtype=np.uint8)
        one_atom_mols = []
        for i in atomic_numbers:
            basis_dict = None
            for basis in basis_list:
                # The basis matters so need to be careful which one to use
                one_atom_mol_tmp = pyscf.gto.M(atom=f"{i} 0 0 0", spin=None, basis=basis)
                if even_tempered:
                    one_atom_mol = build_mol_with_even_tempered_basis(
                        one_atom_mol_tmp, beta=beta, spin=None
                    )
                else:
                    if uncontracted:
                        new_basis = {}
                        for key in one_atom_mol_tmp._basis:
                            new_basis[key] = pyscf.gto.uncontract(one_atom_mol_tmp._basis[key])
                        one_atom_mol = pyscf.gto.M(
                            atom=f"{i} 0 0 0", spin=None, basis=new_basis
                        )
                    else:
                        one_atom_mol = one_atom_mol_tmp
                if basis_dict is None:
                    basis_dict = one_atom_mol._basis
                else:
                    for key in one_atom_mol._basis:
                        basis_dict[key].extend(one_atom_mol._basis[key])
            if add_basis_functions is not None:
                element = pyscf.data.elements.ELEMENTS[i]
                if element in add_basis_functions:
                    basis_dict[element].extend(
                        add_basis_functions[element]
                    )
            one_atom_mol = pyscf.gto.M(
                atom=f"{i} 0 0 0", spin=None, basis=basis_dict
            )
            one_atom_mols.append(one_atom_mol)
        basis_dict, irreps_per_atom, basis_integrals = cls.get_irreps_and_integrals(one_atom_mols)
        return cls(
            atomic_numbers=atomic_numbers,
            basis_dict=basis_dict,
            irreps_per_atom=irreps_per_atom,
            integrals=basis_integrals,
        )
    