import logging
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from typing import List
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import NeptuneLogger

from data_utils.data_dir import DataDir
from validator.validate import (
    validate_and_load_embeddings,
)
from training_pipeline.logger_factory import (
    NeptuneLoggerFactory,
)
from training_pipeline.model import (
    UniversalModel,
)
from training_pipeline.tasks import ValidTasks
from training_pipeline.data_module import (
    BehavioralDataModule,
)
from training_pipeline.constants import (
    BATCH_SIZE,
    MAX_EMBEDDING_DIM,
    HIDDEN_SIZE_THIN,
    HIDDEN_SIZE_WIDE,
    LEARNING_RATE,
    MAX_EPOCH,
)
from training_pipeline.target_data import (
    TargetData,
)
from training_pipeline.task_constructor import (
    TaskConstructor,
    TaskSettings,
    transform_client_ids_and_embeddings,
)
from training_pipeline.metric_aggregator import (
    MetricsAggregator,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def run_training(
    task_settings: TaskSettings,
    embeddings: np.ndarray,
    client_ids: np.ndarray,
    target_data: TargetData,
    num_workers: int,
    accelerator: str,
    devices: List[int] | str | int,
    neptune_logger: NeptuneLogger,
) -> None:
    """
    Function for running the training of a model, with all the training
    parameters already established.

    Args:
        task_settings (TaskSettings): Settings for running the task
        embeddings (np.ndarray): Embeddings to be used as the input to the model
        client_ids (np.ndarray): The ids of clients, in order as their embeddings are in `embeddings`.
        target_data (TargetData): Target purchase data based on which targets are computed
        num_workers (int): Number of workers to be used for loading data
        accelerator (str): Type of device to run training on (e.g. gpu, cpu, etc.)
        devices (List[int] | str | int): id of devices used for training
        neptune_logger (NeptuneLogger): logger instance where training information is logged
    """

    data = BehavioralDataModule(
        embeddings=embeddings,
        client_ids=client_ids,
        target_data=target_data,
        target_calculator=task_settings.target_calculator,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    model = UniversalModel(
        embedding_dim=embeddings.shape[1],
        output_dim=task_settings.target_calculator.target_dim,
        hidden_size_thin=HIDDEN_SIZE_THIN,
        hidden_size_wide=HIDDEN_SIZE_WIDE,
        learning_rate=LEARNING_RATE,
        metric_calculator=task_settings.metric_calculator,
        loss_fn=task_settings.loss_fn,
        metrics_tracker=task_settings.metrics_tracker,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCH,
        logger=neptune_logger,
        callbacks=RichProgressBar(leave=True),
        log_every_n_steps=5000,
    )

    trainer.fit(model=model, datamodule=data)


def run_tasks(
    neptune_logger_factory: NeptuneLoggerFactory,
    tasks: List[ValidTasks],
    task_constructor: TaskConstructor,
    data_dir: DataDir,
    embeddings_dir: Path,
    num_workers: int,
    accelerator: str,
    devices: List[int] | str | int,
    score_dir: Path | None,
    disable_relevant_clients_check: bool,
) -> None:
    """
    Function for running a task, i.e. setting up the training, and the starting the training. This method first
    prepares running paramteres based on preliminary setup, and the calls the `run_train` method.

    Args:
        neptune_logger_factory (NeptuneLoggerFactory): Factory that can generate instance of neptune loggers
            with some pre-set parameters common to the embedding on which experiments are run.
        tasks (List[ValidTasks]): tasks on which the embeddings are to be evaluated.
        task_constructor (TaskConstructor): object for generating training settings based on the task
        data_dir (DataDir): container for simplified access to subdirectories of data_dir.
        embeddings_dir (Path): Path to the directory where the embeddings are stored.
        num_workers (int): number of workers to be used for loading data
        accelerator (str): Type of device to run training on (e.g. gpu, cpu, etc.)
        devices (List[int] | str | int): id of devices used for training
        score_dir (Path | None): Path where results are saved in an easy-to-read format, parallel to netune logging.
        disable_relevant_clients_check (bool): disables validator check for relevant clients
    """
    client_ids, embeddings = validate_and_load_embeddings(
        input_dir=data_dir.input_dir,
        embeddings_dir=embeddings_dir,
        max_embedding_dim=MAX_EMBEDDING_DIM,
        disable_relevant_clients_check=disable_relevant_clients_check,
    )
    target_data = TargetData.read_from_dir(target_dir=data_dir.target_dir)
    metrics_aggregator = MetricsAggregator()
    for task in tasks:
        logger.info("Running on %s", task.value)
        logger.info("Constructing task specific data structures")
        task_settings = task_constructor.construct_task(task=task)

        logger.info("Transforming client ids")
        (
            transformed_client_ids,
            transformed_embeddings,
        ) = transform_client_ids_and_embeddings(
            task=task, client_ids=client_ids, embeddings=embeddings, data_dir=data_dir
        )

        logger.info("Setting up training logger")
        neptune_logger = neptune_logger_factory.get_logger(task=task)

        logger.info("Running training")
        run_training(
            task_settings=task_settings,
            embeddings=transformed_embeddings,
            client_ids=transformed_client_ids,
            target_data=target_data,
            num_workers=num_workers,
            accelerator=accelerator,
            devices=devices,
            neptune_logger=neptune_logger,
        )
        neptune_logger.experiment.stop()

        metrics_aggregator.update(
            task=task, metrics_tracker=task_settings.metrics_tracker
        )
        logger.info("Run on %s completed", task.value)

    if score_dir:
        metrics_aggregator.save(score_dir=score_dir)
