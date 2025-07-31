import logging
from typing import Literal
from pytorch_lightning.loggers import NeptuneLogger, Logger

from training_pipeline.tasks import (
    ValidTasks,
)
from training_pipeline.train_logging_config import (
    TrainLoggingConfig,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class NeptuneLoggerFactory:
    """
    Neptune logger factory based on config params.
    """

    def __init__(
        self,
        project: None | str,
        api_key: None | str,
        name: str,
        train_logging_config: TrainLoggingConfig,
    ):
        """
        Sets up global logger config.

        If the project name is not specified the logger is set up in offline mode. If it is,
        we check if there is an available API key, and if there is, we log into the specified project.
        If there is no API key, we log offline.

        Args:
            project (None | str) : name of neptune project,
            api_key (None | str) : api key
            name (str) : human readable name for the experiment.
            train_logging_config (TrainLoggingConfig): train logging config
        """
        self.mode: str = "async"
        self.project = project
        self.api_key = api_key
        self.name = name
        self.train_logging_config = train_logging_config

        if self.project is None:
            self.mode = "offline"
            logger.warning(
                "No Neptune project is specified, loggin offline. To specify Neptune project, add command line argument --neptune-project workspace/project"
            )
        elif self.api_key is None:
            self.mode = "offline"
            logger.warning(
                "No Neptune API key found, using offline logging. To specify Neptune api key, set --neptune-api-key command line argument."
            )

    def _construct_logger(
        self, task: ValidTasks, **neptune_run_kwargs
    ) -> NeptuneLogger:
        """
        Adds task specific information to logger config and returns logger.

        Args:
            task (ValidTasks): type of the task
        Returns:
            NeptuneLogger : configured neptune logger
        """
        logger_name = (
            f"{self.name}:{self.train_logging_config.logging_task_name(task=task)}"
        )
        return NeptuneLogger(
            mode=self.mode,
            api_key=self.api_key,
            project=self.project,
            name=logger_name,
            log_model_checkpoints=False,
            **neptune_run_kwargs,
        )

    def get_logger(
        self, task: ValidTasks, **neptune_run_kwargs
    ) -> Logger | Literal[False]:
        """
        Method checks logging configuration and returns either logger if enabled or False.
        Args:
            task (ValidTasks): type of the task
            logging_config (LoggingConfig): Logging parameters configuration.
            train_logger (Logger): Training logger.
        """

        return (
            self._construct_logger(task=task, **neptune_run_kwargs)
            if self.train_logging_config.logging_enabled
            else False
        )
