
from typing import Callable, Optional
import numpy as np

from .base import OnlineLearner


class OMD(OnlineLearner):

    def __init__(
        self,
        dim: int,
        step_size: float,
        constraint_set: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        x0: Optional[np.ndarray] = None,
    ):
        super().__init__(dim)

        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.eta = step_size
        self.project = constraint_set

        if x0 is not None:
            if x0.shape != (dim,):
                raise ValueError(f"x0 must have shape ({dim},), got {x0.shape}")
            self.x = x0.copy()
        else:
            self.x = np.zeros(dim)

        self.prev_hint = np.zeros(dim)

        self._x0 = self.x.copy()

    def get_action(self, hint: Optional[np.ndarray] = None) -> np.ndarray:
        if hint is not None:
            hint = np.asarray(hint)
            if hint.shape != (self.dim,):
                raise ValueError(f"hint must have shape ({self.dim},), got {hint.shape}")
            x_opt = self.x - self.eta * hint
            if self.project is not None:
                x_opt = self.project(x_opt)
            return x_opt

        return self.x.copy()

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

        new_hint = np.zeros(self.dim) if hint is None else np.asarray(hint)
        if new_hint.shape != (self.dim,):
            raise ValueError(
                f"hint must have shape ({self.dim},), got {new_hint.shape}"
            )

        effective_gradient = gradient - self.prev_hint + new_hint

        self.x = self.x - self.eta * effective_gradient

        if self.project is not None:
            self.x = self.project(self.x)

        self.prev_hint = new_hint.copy()

    def reset(self) -> None:
        self.x = self._x0.copy()
        self.prev_hint = np.zeros(self.dim)

    def set_iterate(self, x: np.ndarray) -> None:
        x = np.asarray(x)
        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},), got {x.shape}")
        self.x = x.copy()
