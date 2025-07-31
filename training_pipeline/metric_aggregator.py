import json

from pathlib import Path
from typing import Dict, List, Tuple

from training_pipeline.tasks import (
    ValidTasks,
)
from training_pipeline.train_logging_config import (
    TrainLoggingConfig,
)
from training_pipeline.metrics_containers import (
    MetricContainer,
)
from training_pipeline.constants import (
    MAX_EPOCH,
)


class IncorrectMetricsTracker(Exception):
    pass


class MetricsAggregator:
    """
    Class for aggregating metrics collected during training.
    """

    def __init__(self):
        self._aggregated_metrics: Dict[ValidTasks, List[MetricContainer]] = {}

    def update(self, task: ValidTasks, metrics_tracker: List[MetricContainer]) -> None:
        """
        Method for attaching a metric tracker for aggregation later.
        """
        if len(metrics_tracker) != MAX_EPOCH:
            raise IncorrectMetricsTracker(
                f"There are {len(metrics_tracker)} metrics and {MAX_EPOCH} epochs. Number of metrics should be equal to number of epochs."
            )
        self._aggregated_metrics[task] = metrics_tracker

    def _find_best_weighted_metrics_and_epochs(
        self, train_logging_config: TrainLoggingConfig
    ):
        """
        Method for determining max score and corresponding epoch from recorded scores.
        """

        def extract_weighted_metric(
            epoch_and_weighted_metric: Tuple[int, float],
        ) -> float:
            _, weighted_metric = epoch_and_weighted_metric
            return weighted_metric

        self._best_weighted_metrics: Dict[str, float] = {}
        self._best_epochs: Dict[str, int] = {}
        for task, metric_tracker in self._aggregated_metrics.items():
            weighted_metrics = [
                metric_container.compute_weighted_metric()
                for metric_container in metric_tracker
            ]
            best_epoch, best_weighted_metric = max(
                enumerate(weighted_metrics),
                key=extract_weighted_metric,
            )
            self._best_weighted_metrics[
                train_logging_config.logging_task_name(task=task)
            ] = round(best_weighted_metric, 4)
            self._best_epochs[train_logging_config.logging_task_name(task=task)] = (
                best_epoch
            )

    def _save_best_scores(self, score_dir: Path):
        scores_fn = score_dir / "scores.json"
        with open(scores_fn, "w") as scores_file:
            json.dump(self._best_weighted_metrics | {"placeholder": -1}, scores_file)

    def _save_best_epoch(self, score_dir: Path):
        epochs_fn = score_dir / "epochs.json"
        with open(epochs_fn, "w") as epochs_file:
            json.dump(self._best_epochs, epochs_file)

    def save(self, score_dir: Path, train_logging_config: TrainLoggingConfig):
        """
        Method that aggreagates the collected metrics, and saves them.
        """
        self._find_best_weighted_metrics_and_epochs(
            train_logging_config=train_logging_config
        )

        self._save_best_scores(score_dir=score_dir)
        self._save_best_epoch(score_dir=score_dir)
