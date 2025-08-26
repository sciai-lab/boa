import torch

from tensorframes.utils.wigner import (
    euler_angle_inversion,
    euler_angles_yxy,
    wigner_D_from_matrix,
)


class LFrames:
    """Class representing a collection of o3 matrices."""

    def __init__(
        self, matrices: torch.Tensor, cache_wigner: bool = True, spatial_dim: int = 3
    ) -> None:
        """Initialize the LFrames class.

        Args:
            matrices (torch.Tensor): Tensor of shape (..., spatial_dim, spatial_dim) representing the rotation matrices.
            cache_wigner (bool, optional): Whether to cache the Wigner D matrices. Defaults to True.
            spatial_dim (int, optional): Dimension of the spatial vectors. Defaults to 3.

        .. note::
            So far this only supports 3D rotations.
        """
        assert spatial_dim == 3, "So far only 3D rotations are supported."
        assert matrices.shape[-2:] == (
            spatial_dim,
            spatial_dim,
        ), "Rotations must be of shape (..., spatial_dim, spatial_dim)"

        self.matrices = matrices
        self.spatial_dim = spatial_dim

        self._det = None
        self._inv = None
        self._angles = None

        self.cache_wigner = cache_wigner
        self.wigner_cache = {}

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = torch.linalg.det(self.matrices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.matrices.transpose(-1, -2)
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            self._angles = euler_angles_yxy(self.det[:, None, None] * self.matrices)
        return self._angles

    @property
    def shape(self) -> torch.Size:
        """Shape of the o3 matrices.

        Returns:
            torch.Size: Size of the o3 matrices.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the o3 matrices.

        Returns:
            torch.device: Device of the o3 matrices.
        """
        return self.matrices.device

    def inverse_lframes(self) -> "LFrames":
        """Returns the inverse of the LFrames object.

        Returns:
            LFrames: LFrames object containing the inverse rotation matrices.
        """
        return InvLFrames(self)

    def index_select(self, indices: torch.Tensor) -> "LFrames":
        """Selects the rotation matrices corresponding to the given indices.

        Args:
            indices (torch.Tensor): Tensor containing the indices to select.

        Returns:
            LFrames: LFrames object containing the selected rotation matrices.
        """

        return IndexSelectLFrames(self, indices)

    def wigner_D(self, l: int) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        if self.cache_wigner and l in self.wigner_cache:
            return self.wigner_cache[l]
        else:
            wigner = wigner_D_from_matrix(
                l, self.det[:, None, None] * self.matrices, angles=self.angles
            )  # * self.det to ensure wigner gets rotation matrix
            if self.cache_wigner:
                self.wigner_cache[l] = wigner
            return wigner


class InvLFrames(LFrames):
    """Represents the inverse of a collection of o3 matrices."""

    def __init__(self, lframes: LFrames) -> None:
        """Initialize the InvLFrames class.

        Args:
            lframes (LFrames): The LFrames object.

        Returns:
            None
        """
        self.lframes = lframes
        self.spatial_dim = lframes.spatial_dim

        self._det = None
        self._inv = None
        self._angles = None
        self._matrices = None

    @property
    def matrices(self) -> torch.Tensor:
        """Returns the matrices stored in the lframes object.

        Returns:
            torch.Tensor: The matrices stored in the lframes object.
        """
        if self._matrices is None:
            self._matrices = self.lframes.inv
        return self._matrices

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self.lframes.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.lframes.matrices
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            self._angles = euler_angle_inversion(self.lframes.angles)
        return self._angles

    def index_select(self, indices: torch.Tensor) -> LFrames:
        """Selects the rotation matrices corresponding to the given indices.

        Args:
            indices (torch.Tensor): Tensor containing the indices to select.

        Returns:
            LFrames: LFrames object containing the selected rotation matrices.
        """
        return IndexSelectLFrames(self, indices)

    def wigner_D(self, l: int) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        return self.lframes.wigner_D(l).transpose(-1, -2)

    def inverse_lframes(self) -> LFrames:
        """Returns the original reference to the LFrames object."""
        return self.lframes


class IndexSelectLFrames(LFrames):
    """Represents a selection of rotation matrices from an LFrames object.

    The selection is done on the fly.
    """

    def __init__(self, lframes: LFrames, indices: torch.Tensor) -> None:
        """Initialize the IndexSelectLFrames object.

        Args:
            lframes (LFrames): The LFrames object.
            indices (torch.Tensor): The indices.

        Returns:
            None
        """

        self.lframes = lframes
        self.indices = indices
        self.spatial_dim = lframes.spatial_dim

        self._matrices = None
        self._wigner_cache = {}
        self._det = None
        self._inv = None
        self._angles = None

    @property
    def wigner_cache(self) -> dict:
        """Returns the wigner cache for the current instance.

        If the wigner cache has not been initialized, it is initialized by indexing the wigner cache
        attribute of the lframes object with the indices attribute of the current object.

        Returns:
            dict: A dictionary containing the wigner cache.
        """
        if self._wigner_cache == {}:
            for l in self.lframes.wigner_cache:
                self._wigner_cache[l] = self.lframes.wigner_cache[l].index_select(0, self.indices)
        return self._wigner_cache

    @property
    def matrices(self) -> torch.Tensor:
        """Returns the matrices stored in the lframes object.

        If the matrices have not been initialized, they are initialized by indexing the matrices
        attribute of the lframes object with the indices attribute of the current object.

        Returns:
            torch.Tensor: The matrices stored in the lframes object.
        """
        if self._matrices is None:
            self._matrices = self.lframes.matrices.index_select(0, self.indices)
        return self._matrices

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self.lframes.det.index_select(0, self.indices)
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.lframes.inv.index_select(0, self.indices)
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            self._angles = self.lframes.angles.index_select(0, self.indices)
        return self._angles

    def index_select(self, indices: torch.Tensor) -> LFrames:
        """Selects the rotation matrices corresponding to the given indices."""
        indexed_indices = self.indices.index_select(0, indices)
        return IndexSelectLFrames(lframes=self.lframes, indices=indexed_indices)

    def wigner_D(self, l: int) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        if l not in self.wigner_cache:
            # in some cases this may not be the most efficient way since all wigners are calculated.
            self.wigner_cache[l] = self.lframes.wigner_D(l).index_select(0, self.indices)
        return self.wigner_cache[l]

    def inverse_lframes(self) -> LFrames:
        """Returns the original reference to the LFrames object."""
        return InvLFrames(self)


class ChangeOfLFrames(LFrames):
    """Represents a change of frames between two LFrames objects."""

    def __init__(self, lframes_start: LFrames, lframes_end: LFrames) -> None:
        """Initialize the ChangeOfLFrames class.

        Args:
            lframes_start (LFrames): The LFrames object from where to start the transform.
            lframes_end (LFrames): The LFrames object in which to end the transform.
        """
        assert (
            lframes_start.shape == lframes_end.shape
        ), "Both LFrames objects must have the same shape."
        self.lframes_start = lframes_start
        self.lframes_end = lframes_end
        self.matrices = torch.bmm(lframes_end.matrices, lframes_start.inv)
        self.spatial_dim = lframes_start.spatial_dim

        self._det = None
        self._inv = None
        self._angles = None

    @property
    def det(self) -> torch.Tensor:
        """Determinant of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the determinants.
        """
        if self._det is None:
            self._det = self.lframes_start.det * self.lframes_end.det
        return self._det

    @property
    def inv(self) -> torch.Tensor:
        """Inverse of the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the inverses.
        """
        if self._inv is None:
            self._inv = self.matrices.transpose(-1, -2)
        return self._inv

    @property
    def angles(self) -> torch.Tensor:
        """Euler angles in yxy convention corresponding to the o3 matrices.

        Returns:
            torch.Tensor: Tensor containing the Euler angles.
        """
        if self._angles is None:
            # note: these are actually never used, given the definition of wigners below
            self._angles = euler_angles_yxy(self.matrices)
        return self._angles

    @property
    def shape(self) -> torch.Size:
        """Shape of the o3 matrices.

        Returns:
            torch.Size: Size of the o3 matrices.
        """
        return self.matrices.shape

    @property
    def device(self) -> torch.device:
        """Device of the o3 matrices.

        Returns:
            torch.device: Device of the o3 matrices.
        """
        return self.matrices.device

    def wigner_D(self, l: int) -> torch.Tensor:
        """Wigner D matrices corresponding to the rotation matrices.

        Args:
            l (int): Degree of the Wigner D matrices.

        Returns:
            torch.Tensor: Tensor containing the Wigner D matrices.
        """
        return torch.bmm(
            self.lframes_end.wigner_D(l),
            self.lframes_start.wigner_D(l).transpose(-1, -2),
        )

    def inverse_lframes(self) -> "ChangeOfLFrames":
        """Returns the inverse of the ChangeOfLFrames object.

        Returns:
            ChangeOfLFrames: ChangeOfLFrames object containing the inverse rotation matrices.
        """
        return ChangeOfLFrames(lframes_start=self.lframes_end, lframes_end=self.lframes_start)


if __name__ == "__main__":
    # Example usage:
    matrices = torch.rand(2, 3, 3)
    lframes = LFrames(matrices)
    print("wigner_d for l=2:", lframes.wigner_D(2))

    matrices2 = torch.rand(2, 3, 3)
    lframes2 = LFrames(matrices2)
    print("wigner_d for l=2:", lframes2.wigner_D(2))
    print("wigner_d for l=0:", lframes2.wigner_D(0))

    change = ChangeOfLFrames(lframes, lframes2)
    print("wigner_d for l=1:", change.wigner_D(1))
    print("wigner_d for l=2:", change.wigner_D(2))
