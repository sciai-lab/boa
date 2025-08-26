import torch

from tensorframes.lframes.lframes import LFrames


class Reps:
    """A class that represents a tensor representation.

    This class is a template for subclasses that implement specific tensor representations.
    """

    def __init__(self):
        """Initializes a new instance of the class."""

    def __repr__(self) -> str:
        """
        str: Returns a string representation of the `Reps` object.
        """
        assert NotImplementedError, "Subclasses must implement this method."
        return ""

    @property
    def dim(self) -> int:
        """
        int: The total dimension of the `Reps` object.
        """
        assert NotImplementedError, "Subclasses must implement this method."
        return 0

    def get_transform_class(self) -> "RepsTransform":
        """Returns an instance of the `RepsTransform` class based on the `Reps` object.

        Returns:
            RepsTransform: An instance of the `RepsTransform` class.
        """
        assert NotImplementedError, "Subclasses must implement this method."
        return RepsTransform()


class RepsTransform(torch.nn.Module):
    """A class that represents a transformation of a tensor based on a given representation.

    This class is a template for subclasses that implement specific transformations based on a
    given representation.
    """

    def __init__(self):
        """Initializes a new instance of the class."""
        assert NotImplementedError, "Subclasses must implement this method."

    def forward(self, x: torch.Tensor, basis_change: LFrames) -> torch.Tensor:
        """Applies the transformation to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            basis_change (LFrames): The basis change to apply to the tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        assert NotImplementedError, "Subclasses must implement this method."
        return x
