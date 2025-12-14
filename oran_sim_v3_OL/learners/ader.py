
from typing import Callable, Optional, List
import numpy as np

from .base import OnlineLearner
from .omd import OMD


class ADER(OnlineLearner):

    def __init__(
        self,
        dim: int,
        T: int,
        D: float,
        L: float,
        constraint_set: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        x0: Optional[np.ndarray] = None,
    ):
        super().__init__(dim)

        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if D <= 0:
            raise ValueError(f"D must be positive, got {D}")
        if L <= 0:
            raise ValueError(f"L must be positive, got {L}")

        self.T = T
        self.D = D
        self.L = L
        self.project = constraint_set

        self.N = 1 + int(np.ceil(np.log2(np.sqrt(4 + 8 * T))))

        sqrt_T = np.sqrt(T)
        self.step_sizes: List[float] = [
            D * (2 ** (i - 1)) / (L * sqrt_T)
            for i in range(1, self.N + 1)
        ]

        self.beta = np.sqrt(2 * np.log(self.N)) / (L * D * sqrt_T)

        self.weights = np.ones(self.N) / self.N

        x_init = x0 if x0 is not None else np.zeros(dim)
        if x0 is not None and x0.shape != (dim,):
            raise ValueError(f"x0 must have shape ({dim},), got {x0.shape}")

        self.experts: List[OMD] = []
        for eta in self.step_sizes:
            expert = OMD(dim=dim, step_size=eta, constraint_set=constraint_set)
            expert.x = x_init.copy()
            self.experts.append(expert)

        self._x0 = x_init.copy()
        self._initial_weights = self.weights.copy()

    def get_action(self, hint: Optional[np.ndarray] = None) -> np.ndarray:
        actions = np.array([expert.get_action() for expert in self.experts])

        return np.average(actions, weights=self.weights, axis=0)

    def update(
        self,
        gradient: np.ndarray,
        hint: Optional[np.ndarray] = None
    ) -> None:
        gradient = np.asarray(gradient)
        if gradient.shape != (self.dim,):
            raise ValueError(
                f"gradient must have shape ({self.dim},), got {gradient.shape}"
            )

        losses = np.array([
            np.dot(gradient, expert.get_action())
            for expert in self.experts
        ])

        for expert in self.experts:
            expert.update(gradient)

        self.weights = self.weights * np.exp(-self.beta * losses)

        self.weights = np.maximum(self.weights, 1e-300)

        self.weights = self.weights / self.weights.sum()

    def reset(self) -> None:
        self.weights = self._initial_weights.copy()
        for expert in self.experts:
            expert.x = self._x0.copy()
            expert.prev_hint = np.zeros(self.dim)

    def get_expert_weights(self) -> np.ndarray:
        return self.weights.copy()

    def get_expert_actions(self) -> np.ndarray:
        return np.array([expert.get_action() for expert in self.experts])
