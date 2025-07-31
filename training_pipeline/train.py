import argparse
import logging

from itertools import chain

from pathlib import Path

from training_pipeline.tasks import (
    ChurnTasks,
    PropensityTasks,
    parse_task,
)
from training_pipeline.task_constructor import (
    TaskConstructor,
)
from training_pipeline.logger_factory import (
    NeptuneLoggerFactory,
)
from training_pipeline.train_runner import (
    run_tasks,
)
from data_utils.data_dir import DataDir
from training_pipeline.train_logging_config import (
    TrainLoggingConfig,
)
from typing import List

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        required=True,
        help="Directory where input embeddings are stored",
    )
    valid_tasks = " ".join([task.value for task in chain(ChurnTasks, PropensityTasks)])
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        required=True,
        help=f"Name of the task to train out of: {valid_tasks}",
    )
    parser.add_argument("--log-name", type=str, required=True, help="Experiment name")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )

    parser.add_argument(
        "--devices",
        nargs="*",
        required=True,
        type=str,
        help='List of devices to use. Possible options: "auto", id of single device to use or list of ids of devices to use.',
    )

    parser.add_argument(
        "--neptune-api-token",
        required=False,
        type=str,
        help="Neptune API token.",
    )

    parser.add_argument(
        "--neptune-project",
        required=False,
        type=str,
        help="Name of Neptune project within workspace to save result to.",
    )

    parser.add_argument(
        "--score-dir",
        required=False,
        type=str,
        help="Path to directory where to save best scores for each task",
    )
    parser.add_argument(
        "--disable-relevant-clients-check",
        action="store_true",
        help="Disables relevant clients check in validator, but enables embeddings for sets of clients other than relevant clients.",
    )
    parser.add_argument(
        "--hidden-logging-mode",
        action="store_true",
        help="Disables progress bar, model summary and logger. Overwrites tasks names to hidden ones",
    )
    return parser


def parse_devices(device_arg: List[str]) -> List[int] | int | str:
    """
    Method to parse --devices argument of argparse and return devices to use.
    Args:

        device_arg (List[str]): --devices command line argument from argparse

    Returns:
        List[int] | int | str : devices to use: "auto", a single device id or list of device ids.
    """
    if (len(device_arg) == 1) and (device_arg[0] == "auto"):
        return "auto"
    else:
        try:
            return [int(device) for device in device_arg]
        except ValueError:
            raise ValueError(
                f'Devices argument should be one one of "auto", int or list of ints, received: "{" ".join(device_arg)}"'
            )


def main(params) -> None:
    tasks = [parse_task(task) for task in params.tasks]
    train_logging_config = TrainLoggingConfig(
        hidden_logging_mode=params.hidden_logging_mode
    )
    neptune_logger_factory = NeptuneLoggerFactory(
        project=params.neptune_project,
        api_key=params.neptune_api_token,
        name=params.log_name,
        train_logging_config=train_logging_config,
    )

    data_dir = DataDir(data_dir=Path(params.data_dir))
    task_constructor = TaskConstructor(data_dir=data_dir)
    score_dir = Path(params.score_dir) if params.score_dir else None

    run_tasks(
        train_logging_config=train_logging_config,
        neptune_logger_factory=neptune_logger_factory,
        tasks=tasks,
        task_constructor=task_constructor,
        data_dir=data_dir,
        embeddings_dir=Path(params.embeddings_dir),
        num_workers=params.num_workers,
        accelerator=params.accelerator,
        devices=parse_devices(params.devices),
        score_dir=score_dir,
        disable_relevant_clients_check=params.disable_relevant_clients_check,
    )


if __name__ == "__main__":
    parse = get_parser()
    params = parse.parse_args()

    main(params)
