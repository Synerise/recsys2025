import numpy as np
import torch
import torch.nn.functional as F

from typing import Callable, List, Tuple
from dataclasses import dataclass, field

from training_pipeline.metric_calculators import (
    MetricCalculator,
    ChurnMetricCalculator,
    PropensityMetricCalculator,
)
from training_pipeline.tasks import (
    ValidTasks,
    ChurnTasks,
    PropensityTasks,
    TaskNotSupportedError,
)
from training_pipeline.target_calculators import (
    TargetCalculator,
    ChurnTargetCalculator,
    PropensityTargetCalculator,
)

from data_utils.data_dir import DataDir
from training_pipeline.metrics_containers import (
    MetricContainer,
)


@dataclass(frozen=True)
class TaskSettings:
    """
    Container class which stores all task specific data structures.
    """

    target_calculator: TargetCalculator
    metric_calculator: MetricCalculator
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    metrics_tracker: List[MetricContainer] = field(default_factory=list)


class TaskConstructor:
    """
    Class for constructing all task specific data structures.
    """

    def __init__(self, data_dir: DataDir):
        """
        Args:
            data_dir (DataDir): container for simplified access to subdirectories of data_dir.
        """
        self.data_dir = data_dir

    def construct_task(self, task: ValidTasks) -> TaskSettings:
        """
        Method for constructing task specific data structures.

        Args:
            task (ValidTasks): task for which data structures are constructed.
        Returns:
            TaskSettings: container with data structures for given task
        """
        if isinstance(task, ChurnTasks):
            return self._construct_churn_task(task=task)
        elif isinstance(task, PropensityTasks):
            return self._construct_propensity_task(task=task)
        else:
            raise TaskNotSupportedError(f"Task: {task.value} is not supported")

    def _construct_churn_task(self, task: ChurnTasks) -> TaskSettings:
        target_calculator = ChurnTargetCalculator()
        metric_calculator = ChurnMetricCalculator()
        return TaskSettings(
            target_calculator=target_calculator,
            metric_calculator=metric_calculator,
            loss_fn=F.binary_cross_entropy_with_logits,
        )

    def _construct_propensity_task(self, task: PropensityTasks) -> TaskSettings:
        propensity_targets, popularity_data = self._load_propensity_targets(task)

        target_calculator = PropensityTargetCalculator(
            task=task, propensity_targets=propensity_targets
        )

        metric_calculator = PropensityMetricCalculator(
            output_dim=target_calculator.target_dim,
            popularity_data=popularity_data,
        )

        return TaskSettings(
            target_calculator=target_calculator,
            metric_calculator=metric_calculator,
            loss_fn=F.binary_cross_entropy_with_logits,
        )

    def _load_propensity_targets(
        self,
        task: PropensityTasks,
    ) -> Tuple[np.ndarray, np.ndarray]:

        propensity_targets = np.load(
            self.data_dir.target_dir / f"{task.value}.npy",
            allow_pickle=True,
        )
        popularity_data = np.load(
            self.data_dir.target_dir / f"popularity_{task.value}.npy"
        )
        return propensity_targets, popularity_data


def transform_client_ids_and_embeddings(
    task: ValidTasks,
    client_ids: np.ndarray,
    embeddings: np.ndarray,
    data_dir: DataDir,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restrict client_ids to active clients for churn task.
    """
    if task == ChurnTasks.CHURN:
        active_clients = np.load(data_dir.target_dir / "active_clients.npy")
        mask = np.isin(client_ids, active_clients)
        return client_ids[mask], embeddings[mask]
    return client_ids, embeddings
