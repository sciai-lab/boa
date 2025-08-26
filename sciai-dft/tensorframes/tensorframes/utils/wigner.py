import torch

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
# _Jd is a list of tensors of shape (2l+1, 2l+1)
# We don't load during import time since this causes problems with pytorch lightnings checkpoint loading
_Jd = None


def acos_grad(x: torch.Tensor) -> torch.Tensor:
    """Compute the gradient of the arccos function.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The gradient of the arccos function.
    """
    return -1.0 / torch.sqrt(1.0 - x**2)


class safe_acos(torch.autograd.Function):
    """A custom autograd function that computes the inverse cosine of the input tensor while
    protecting against NaN outputs and large gradients.

    adapted from https://github.com/pytorch/pytorch/issues/8069

    Args:
        input (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with inverse cosine values.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Applies the forward pass of the Wigner activation function.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): The context object for autograd.
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Wigner activation function.
        """
        ctx.save_for_backward(input)

        # protect ourselves from nan outputs in forward pass.
        return torch.clamp(input, min=-1 + 1e-6, max=1 - 1e-6).acos()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Computes the backward pass for the Wigner function.

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): The context object.
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input.
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        # protect ourselves from large gradients in backward pass.
        # outside of (-1 + epsilon, 1 - epsilon), gradient value is fixed constant to acos'(1-epsilon)
        epsilon = 0.05
        safe_input = torch.clamp(input, min=-1 + epsilon, max=1 - epsilon)

        return acos_grad(safe_input) * grad_input


def euler_angles_yxy(
    matrix: torch.Tensor, handle_special_cases: bool = False, eps: float = 1e-9
) -> torch.Tensor:
    """Calculate the Euler angles using the yxy convention to match the wigner d from e3nn.

    Args:
        matrix (torch.Tensor): The input rotation matrix of shape (..., 3, 3).
        handle_special_cases (bool, optional): Whether to handle special cases where the matrix is diagonal. Defaults to False.
        eps (float, optional): The epsilon value to use for detecting the special cases. Defaults to 1e-9.

    Returns:
        torch.Tensor: The Euler angles alpha, beta, and gamma, each of shape (...).

    References:
        - https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
    """
    angles = torch.zeros(matrix.shape[:-1], dtype=matrix.dtype, device=matrix.device)

    denominator = torch.where(matrix[..., 2, 1].abs() < eps, eps, matrix[..., 2, 1])

    angles[..., 0] = torch.arctan2(matrix[..., 0, 1], denominator)
    angles[..., 1] = safe_acos.apply(matrix[..., 1, 1])

    denominator = torch.where(matrix[..., 1, 2].abs() < eps, eps, matrix[..., 1, 2])

    angles[..., 2] = torch.arctan2(matrix[..., 1, 0], -denominator)

    if handle_special_cases:
        # hard code diagonal special cases (these do not happen much in practice):
        mask_0 = (matrix - torch.eye(3)).abs().max(-1).values.max(-1).values < eps
        mask_1 = (
            matrix - torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        mask_2 = (
            matrix - torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        mask_3 = (
            matrix - torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        ).abs().max(-1).values.max(-1).values < eps
        angles[mask_0] = 0.0
        angles[mask_1] = torch.tensor([0.0, torch.pi, 0.0])
        angles[mask_2] = torch.tensor([0.0, 0.0, torch.pi])
        angles[mask_3] = torch.tensor([0.0, torch.pi, torch.pi])

    return angles


def euler_angle_inversion(angles: torch.Tensor) -> torch.Tensor:
    """Inverts the given Euler angles to find the euler angles for the transpose o(3) matrix in the
    convention above..

    Args:
        angles (torch.Tensor): A tensor of Euler angles. The shape of the tensor
                               should be (..., 3), where the last dimension represents
                               the three Euler angles.

    Returns:
        torch.Tensor: A tensor of the same shape as `angles` with the first and third
                      angles inverted.
    """
    inv_angles = angles.clone()
    inv_angles[..., 0] = torch.pi - angles[..., 2]
    inv_angles[..., 2] = torch.pi - angles[..., 0]
    return inv_angles


def wigner_D_from_matrix(
    l: int,
    matrix: torch.Tensor,
    angles: torch.Tensor | None = None,
    handle_special_cases: bool = False,
) -> torch.Tensor:
    """Calculate the Wigner D-matrix for a given angular momentum `l` and rotation matrix `matrix`.

    Args:
        l (int): The angular momentum quantum number.
        matrix (torch.Tensor): The rotation matrix. shape (..., 3, 3)
        angles (torch.Tensor, optional): The Euler angles in yxy convention. Of shape (N,3). If not provided, it will be calculated from the matrix. Defaults to None.
        handle_special_cases (bool, optional): Whether to handle special cases where the matrix is diagonal. Defaults to False.

    Returns:
        torch.Tensor: The resulting Wigner D matrix of shape (..., 2l+1, 2l+1)
    """
    if l == 0:
        return torch.ones(matrix.shape[:-2] + (1, 1), dtype=matrix.dtype, device=matrix.device)
    if l == 1:
        return matrix
    else:
        raise NotImplementedError("l > 1 not implemented")
