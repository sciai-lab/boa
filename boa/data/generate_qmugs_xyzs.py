from pathlib import Path

import polars
import pyscf
from loguru import logger

from mldft.datagen.datasets.qmugs import QMUGS


class QMUGSLargeBinsBoa(QMUGS):
    """A subset of QMUGS containing molecules larger than 15 heavy atoms.

    50 molecules from each bin of heavy atoms are randomly (but deterministically with seed 1)
    sampled.
    """

    def __init__(
        self,
        raw_data_dir: str,
        kohn_sham_data_dir: str,
        label_dir: str,
        filename: str,
        name: str = "QMUGS",
        num_processes: int = 1,
        n_mol_per_bin: int = 1,
        seed: int = 1,
    ):
        """Initialize the QMUGS dataset.

        Args:
            raw_data_dir: Path to the raw data directory.
            kohn_sham_data_dir: Path to the kohn-sham data directory.
            label_dir: Path to the label directory.
            filename: Filename to use for the output files.
            name: Name of the dataset.
            num_processes: Number of processes to use for dataset verifying or loading.
        """
        super().__init__(
            raw_data_dir=raw_data_dir,
            kohn_sham_data_dir=kohn_sham_data_dir,
            label_dir=label_dir,
            filename=filename,
            name=name,
            num_processes=num_processes,
        )
        # Remove bins smaller than 16 heavy atoms
        self.structure_info = self.structure_info.filter(polars.col("bin") >= 1)
        # Sample n_mol_per_bin molecules from each bin
        self.n_mol_per_bin = n_mol_per_bin
        self.seed = seed
        self.structure_info = self.structure_info.group_by("bin").map_groups(
            lambda df: df.sample(self.n_mol_per_bin, seed=self.seed)
        )
        # Sort again for reproducibility
        self.structure_info = self.structure_info.sort("id")
        self.num_molecules = self.get_num_molecules()
        # Write in ID file or check if it exists
        id_file = self.kohn_sham_data_dir.parent / "QMUGSLargeBinsBoa.csv"
        if id_file.exists():
            logger.info(f"ID file {id_file.name} already exists, checking if its the same.")
            assert polars.read_csv(id_file).equals(self.structure_info.drop("atomic_numbers")), (
                "The ID file already exists, but it is not the same as the current dataset."
            )
        else:
            self.structure_info.drop("atomic_numbers").write_csv(id_file)

    def write_to_xyz(self, output_dir: str):
        """Write the dataset to XYZ files.

        Args:
            output_dir: Directory to write the XYZ files to.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for idx in self.get_ids():
            atomic_numbers, coordinates = self.load_charges_and_positions(idx)
            num_atoms = len(atomic_numbers)
            xyz_lines = [str(num_atoms), f"{idx}"]
            for atomic_number, (x, y, z) in zip(atomic_numbers, coordinates):
                element = pyscf.data.elements._symbol(atomic_number)
                xyz_lines.append(f"{element} {x} {y} {z}")

            xyz_content = "\n".join(xyz_lines)
            with open(output_path / f"{idx}.xyz", "w") as f:
                f.write(xyz_content)

            logger.info(f"Wrote XYZ file for molecule ID {idx} to {output_path / f'{idx}.xyz'}")


if __name__ == "__main__":
    dataset = QMUGSLargeBinsBoa(
        raw_data_dir="/export/scratch/ialgroup/dft_data/QMUGS/raw",
        kohn_sham_data_dir="/export/scratch/ialgroup/dft_data/QMUGS/kohn_sham",
        label_dir="/export/scratch/ialgroup/dft_data/QMUGS/labels",
        filename="qmugs_xyzs",
        num_processes=4,
    )
    dataset.write_to_xyz(output_dir="/export/scratch/ialgroup/dft_data/QMUGS/xyzs")
