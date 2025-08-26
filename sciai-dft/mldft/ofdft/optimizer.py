import functools
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Type

import torch
from tqdm import tqdm

from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ofdft.energies import Energies


class Optimizer(ABC):
    """Base class for optimization algorithms for density optimization."""

    @abstractmethod
    def optimize(
        self,
        sample: OFData,
        energy_functional: Callable[[OFData], tuple[Energies, torch.Tensor]],
        callback: Callable | None = None,
    ) -> Energies:
        """Perform density optimization.

        Args:
            sample: The OFData containing the initial coefficients.
            energy_functional: Callable which returns the energy and gradient vector.
            callback: Optional callback function.

        Returns:
            Final energy.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a string representation of the optimizer."""
        name = self.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{name}({settings})"


def get_pbar_str(sample: OFData, energy: Energies, gradient_norm: float) -> str:
    """Return a string for the tqdm progress bar.

    Args:
        sample: The OFData containing the current coefficients. If the ground state energy is
            available, the energy difference to the ground state is calculated.
        energy: The current energy.
        gradient_norm: The norm of the gradient vector.

    Returns:
        A string for the tqdm progress bar.
    """
    if "of_labels/energies/ground_state_e_electron" in sample:
        ground_state_electronic_energy = sample["of_labels/energies/ground_state_e_electron"]
        if isinstance(ground_state_electronic_energy, torch.Tensor):
            ground_state_electronic_energy = ground_state_electronic_energy.item()
        delta_e = energy.electronic_energy - ground_state_electronic_energy
        delta_e *= 1e3  # convert to mHa
        return f"ΔE={delta_e:.3e} mHa, grad_norm={gradient_norm:.3e}"
    else:
        return f"E_elec={energy.electronic_energy:.6f} Ha, grad_norm={gradient_norm:.2g}"


class GradientDescent(Optimizer):
    """Simple gradient descent optimizer."""

    def __init__(self, learning_rate: float, convergence_tolerance: float, max_cycle: int):
        """Initialize the gradient descent optimizer.

        Args:
            max_cycle: Maximum number of optimization cycles.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            learning_rate: The learning rate.
        """
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle

    def optimize(
        self, sample: OFData, energy_functional: Callable, callback: Callable | None = None
    ) -> Energies:
        """Perform gradient descent optimization."""
        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=int(os.getenv("DENOP_PID", 0)),
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            gradient_norm = torch.norm(projected_gradient).item()

            pbar.set_description(get_pbar_str(sample, energy, gradient_norm))
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                learning_rate = self.learning_rate
                callback(locals())

            if gradient_norm < self.convergence_tolerance:
                break

            sample.coeffs -= self.learning_rate * projected_gradient
        pbar.close()
        return energy


class TorchOptimizer(Optimizer):
    """Wrapper for torch optimizers to be used in the optimization loop."""

    def __init__(
        self,
        torch_optimizer: Type[torch.optim.Optimizer],
        convergence_tolerance: float,
        max_cycle: int,
        **optimizer_kwargs,
    ):
        """Initialize the torch optimizer.

        Args:
            torch_optimizer: The torch optimizer to use. To be able to apply the optimizer with
                hydra, the class is partially applied without any arguments.
            convergence_tolerance: Optimization stops if the gradient norm is below this value.
            max_cycle: Maximum number of optimization cycles.
            optimizer_kwargs: Additional keyword arguments for the optimizer.
        """
        self.torch_optimizer = torch_optimizer
        self.convergence_tolerance = convergence_tolerance
        self.max_cycle = max_cycle
        self.optimizer_kwargs = optimizer_kwargs

    def optimize(
        self, sample: OFData, energy_functional: Callable, callback: Callable | None = None
    ):
        """Optimization loop for a torch optimizer."""
        parameters = sample.coeffs.clone()
        optimizer = self.torch_optimizer([parameters], **self.optimizer_kwargs)

        for cycle in (
            pbar := tqdm(
                range(self.max_cycle),
                leave=False,
                dynamic_ncols=True,
                position=int(os.getenv("DENOP_PID", 0)),
            )
        ):
            energy, gradient_vector = energy_functional(sample)
            projected_gradient = project_gradient(gradient_vector, sample)
            gradient_norm = torch.norm(projected_gradient).item()

            pbar.set_description(get_pbar_str(sample, energy, gradient_norm))
            if callable(callback):
                coeffs = sample.coeffs  # for callback
                if "lr" in self.optimizer_kwargs:
                    learning_rate = self.optimizer_kwargs["lr"]
                else:
                    learning_rate = 0
                callback(locals())

            optimizer.zero_grad()
            parameters.grad = projected_gradient
            optimizer.step()
            update = parameters - sample.coeffs
            # projecting the update is important for, e.g., the Adam optimizer
            sample.coeffs += project_gradient(update, sample)
            # update parameters but clone to not update sample.coeffs in optimizer step
            parameters.data = sample.coeffs.detach().clone()

            if gradient_norm < self.convergence_tolerance:
                break
        pbar.close()
        return energy

    def __str__(self) -> str:
        """Return a string representation of the optimizer.

        This method is overwritten since the torch optimizer can either be a class via direct
        instantiation or via hydra with a partial.
        """
        if isinstance(self.torch_optimizer, functools.partial):  # if called with config
            name = self.torch_optimizer.func.__name__
        else:
            name = self.torch_optimizer.__class__.__name__
        settings = ", ".join(f"{k}={v}" for k, v in self.optimizer_kwargs.items())
        return f"{name}({settings})"
