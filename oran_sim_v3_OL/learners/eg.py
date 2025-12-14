
from typing import Optional
import numpy as np

from .base import OnlineLearner


class EG(OnlineLearner):

    def __init__(
        self,
        n_arms: int,
        step_size: float = 0.1,
        min_weight: float = 1e-8,
    ):
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if min_weight <= 0:
            raise ValueError(f"min_weight must be positive, got {min_weight}")

        super().__init__(dim=n_arms)

        self.n_arms = n_arms
        self.eta = step_size
        self.min_weight = min_weight

        self.weights = np.ones(n_arms)

        self._initial_weights = self.weights.copy()

    def get_action(self, hint: Optional[np.ndarray] = None) -> np.ndarray:
        return self.get_distribution()

    def get_distribution(self) -> np.ndarray:
        total = self.weights.sum()
        if total > 0:
            return self.weights / total
        else:
            return np.ones(self.n_arms) / self.n_arms

    def update(
        self,
        gradient: np.ndarray,
        hint: Optional[np.ndarray] = None
    ) -> None:
        gradient = np.asarray(gradient)
        if gradient.shape != (self.n_arms,):
            raise ValueError(
                f"gradient must have shape ({self.n_arms},), got {gradient.shape}"
            )

        clipped_gradient = np.clip(gradient, -50 / self.eta, 50 / self.eta)

        self.weights = self.weights * np.exp(-self.eta * clipped_gradient)

        self.weights = np.maximum(self.weights, self.min_weight)

    def add_arm(self, initial_weight: float = 1.0) -> None:
        if initial_weight <= 0:
            raise ValueError(f"initial_weight must be positive, got {initial_weight}")

        self.weights = np.append(self.weights, initial_weight)
        self.n_arms += 1
        self.dim = self.n_arms

    def remove_arm(self, index: int) -> None:
        if self.n_arms <= 1:
            raise ValueError("Cannot remove arm: at least one arm must remain")
        if index < 0 or index >= self.n_arms:
            raise ValueError(f"index {index} out of bounds for {self.n_arms} arms")

        self.weights = np.delete(self.weights, index)
        self.n_arms -= 1
        self.dim = self.n_arms

    def reset(self) -> None:
        self.weights = np.ones(self.n_arms)

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()
