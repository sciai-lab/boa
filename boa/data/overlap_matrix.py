import torch


class OverlapMatrix(torch.Tensor):
    """Custom tensor class for the overlap matrix.

    This class is used to control the collation of the overlap matrix in the collate function of
    the OFLoader.
    """

    @staticmethod
    def zero_pad_matrices(matrices):
        """Zero-pad the matrices to the maximum size with ones on the diagonal."""
        max_size = max([matrix.shape[0] for matrix in matrices])
        padded_matrices = []
        for matrix in matrices:
            padded_matrix = torch.eye(max_size, dtype=matrix.dtype, device=matrix.device)
            padded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
            padded_matrices.append(padded_matrix)
        return padded_matrices
