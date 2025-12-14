
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class OnlineLearner(ABC):

    def __init__(self, dim: int):
        self.dim = dim

    @abstractmethod
    def get_action(self, hint: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, gradient: np.ndarray, hint: Optional[np.ndarray] = None) -> None:
        pass

    def reset(self) -> None:
        pass
