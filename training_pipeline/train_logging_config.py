from dataclasses import dataclass
from pytorch_lightning.callbacks import RichProgressBar

from training_pipeline.tasks import (
    ValidTasks,
    ChurnTasks,
    PropensityTasks,
)


@dataclass
class TrainLoggingConfig:
    """
    Class storing logging configuration for basic and hidden modes.
    """

    hidden_logging_mode: bool

    @property
    def logging_enabled(self) -> bool:
        return not self.hidden_logging_mode

    @property
    def callbacks(self) -> RichProgressBar | None:
        if self.hidden_logging_mode:
            return None
        return RichProgressBar(leave=True)

    def logging_task_name(self, task: ValidTasks) -> str:
        safe_task_names = {
            ChurnTasks.CONVERSION: "hidden1",
            PropensityTasks.PROPENSITY_NEW_SKU: "hidden2",
            PropensityTasks.PROPENSITY_PRICE: "hidden3",
        }
        if self.hidden_logging_mode and (task in safe_task_names.keys()):
            return safe_task_names[task]
        return task.value
