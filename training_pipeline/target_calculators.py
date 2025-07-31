import numpy as np
import pandas as pd


from abc import ABC, abstractmethod, abstractproperty
from training_pipeline.tasks import (
    PropensityTasks,
    get_propensity_column,
)


class TargetCalculator(ABC):
    @abstractproperty
    def target_dim(self) -> int:
        """
        Dimensionality of targets.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        """
        Computes targets.

        Args:
            client_id (int): client id
            target_df (pd.DataFrame): target dataset
        """
        raise NotImplementedError


class ChurnTargetCalculator(TargetCalculator):
    @property
    def target_dim(self) -> int:
        return 1

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)
        target[0] = 1 if target_df.loc[target_df["client_id"] == client_id].empty else 0
        return target


class PropensityTargetCalculator(TargetCalculator):
    def __init__(self, task: PropensityTasks, propensity_targets: np.ndarray):
        """
        Args:
            task (PropensityTask): propensity task for target calculator
            propensity_targets (np.ndarray): propensity targets.
        """

        self._propensity_type = get_propensity_column(task=task)
        self._propensity_targets = propensity_targets

    @property
    def target_dim(self) -> int:
        return len(self._propensity_targets)

    def compute_target(self, client_id: int, target_df: pd.DataFrame) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=np.float32)

        cats_in_target = target_df.loc[target_df["client_id"] == client_id][
            self._propensity_type
        ].unique()
        target[
            np.isin(self._propensity_targets, cats_in_target, assume_unique=True)
        ] += 1
        return target
