import numpy as np
import logging

from pathlib import Path
from typing import Optional, List

from validator.utils import (
    _err_msg_if_checks_fail,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class EmbeddingsValidator:
    """
    Validates embeddings and client_ids.
    """

    def __init__(
        self,
        input_dir: Path,
        client_ids: np.ndarray,
        embeddings: np.ndarray,
        max_embeddings_dim: int,
        disable_relevant_clients_check: bool,
    ):
        """
        Initializes embeddings validator instance.

        Args:
            max_embeddings_dim (int): maximal allowed dimension of embeddings
            input_dir (Path): path to directory with input data
            client_ids (np.ndarray): client_ids array
            embeddings (np.ndarray): embeddings array
            disable_relevant_clients_check (bool): if set disables relevant clients
        """
        self._max_embeddings_dim = max_embeddings_dim
        self._input_dir = input_dir
        self._client_ids = client_ids
        self._embeddings = embeddings
        self._relevant_client_ids: Optional[np.ndarray] = (
            self._load_relevant_client_ids()
            if not disable_relevant_clients_check
            else None
        )

    def _load_relevant_client_ids(self) -> np.ndarray:
        return np.load(self._input_dir / "relevant_clients.npy")

    def _err_msg_if_client_ids_have_incorrect_shape(self) -> Optional[str]:
        """
        Checks if client_ids are one dimensional.

        Returns:
            Optional[str]: optional string reporting that client_ids have incorrect shape
        """
        if (client_ids_shape := len(self._client_ids.shape)) != 1:
            return f"client ids are {client_ids_shape} dimensional – should be one dimensional"
        return None

    def _err_msg_if_client_ids_have_incorrect_dtype(self) -> Optional[str]:
        if (client_ids_dtype := self._client_ids.dtype) != np.int64:
            return f"client_ids dtype is {client_ids_dtype} – should be int64"
        return None

    def _err_msg_if_client_ids_coincide_with_all_clients(self) -> Optional[str]:
        """
        Checks client_ids coinsides with all clients in event types.

        Returns:
            Optional[str]: optional string reporting that some clients are missing.
        """
        if isinstance(self._relevant_client_ids, np.ndarray):
            same_length = len(self._relevant_client_ids) == len(self._client_ids)
            inclusion = np.all(np.isin(self._relevant_client_ids, self._client_ids))
            if not (same_length and inclusion):
                return "client ids are not a permutation of relevant clients"
        else:
            logger.warning(
                "Validator will not check if the content of client_ids.npy matches with the list of relevant clients. Embeddings may not conform to competition format."
            )
        return None

    def _err_msg_if_embeddings_have_incorrect_type(self) -> Optional[str]:
        """
        Checks if embeddings have correct type.

        Returns:
            Optional[str]: optional string reporting that embeddings have incorrect type
        """
        if not isinstance(self._embeddings, np.ndarray):
            return f"embeddings have incorrect type: {type(self._embeddings).__name__}. Type should be {np.ndarray.__name__}"
        return None

    def _err_msg_if_embeddings_have_incorrect_shape(self) -> Optional[str]:
        """
        Checks if embeddings have correct shape.

        Returns:
            Optional[str]: optional string reporting that embeddings have incorrect shape.
        """
        if len(self._embeddings.shape) != 2:
            return "embeddings have incorrect shape"
        return None

    def _err_msg_if_embeddings_have_incorrect_dim(self) -> Optional[str]:
        """
        Checks if embeddings have dimension which not exceeds embedding_dim.

        Return:
            Optional[str]: optional string reporting that maximal embeddings dim is exceeded.
        """
        if self._embeddings.shape[1] > self._max_embeddings_dim:
            return f"maximum embeddings dimensionality i.e. {self._max_embeddings_dim} has been exceeded"
        return None

    def _err_msg_if_embeddings_have_incorrect_dtype(self) -> Optional[str]:
        """
        Check if embeddings have correct dtype.
        """
        if self._embeddings.dtype != np.float16:
            return f"embeddings dtype should be float16 instead got {self._embeddings.dtype}"
        return None

    def _err_msg_if_embeddings_and_client_ids_have_different_length(
        self,
    ) -> Optional[str]:
        """
        Checks if embeddings and client_ids have different length.

        Returns:
            Optional[str]: optional string reporting that embeddings and clients_ids have different length.
        """
        if self._embeddings.shape[0] != len(self._client_ids):
            return "client ids and embeddings have different lengths"
        return None

    def _err_msg_if_client_ids_incorrect(self) -> Optional[str]:
        """
        Runs all checks for client_ids.

        Returns:
            Optional[str]: optional string reporting the first check that fails.
        """
        return _err_msg_if_checks_fail(
            err_msg_if_check_fail_fns=[
                self._err_msg_if_client_ids_have_incorrect_shape,
                self._err_msg_if_client_ids_have_incorrect_dtype,
                self._err_msg_if_client_ids_coincide_with_all_clients,
            ],
            success_msg="client ids are valid",
        )

    def _err_msg_if_embeddings_incorrect(self) -> Optional[str]:
        """
        Runs all checks for embeddings.

        Returns:
            Optional[str]: optional string reporting the first check that fails.
        """
        return _err_msg_if_checks_fail(
            err_msg_if_check_fail_fns=[
                self._err_msg_if_embeddings_have_incorrect_type,
                self._err_msg_if_embeddings_have_incorrect_shape,
                self._err_msg_if_embeddings_have_incorrect_dim,
                self._err_msg_if_embeddings_have_incorrect_dtype,
            ],
            success_msg="embeddings are valid",
        )

    def _err_msg_if_inconsistent(self) -> Optional[str]:
        """
        Runs all checks for consistency of embeddings and client_ids.

        Returns:
            Optional[str]: optional string reporting the first check that fails.
        """
        return _err_msg_if_checks_fail(
            err_msg_if_check_fail_fns=[
                self._err_msg_if_embeddings_and_client_ids_have_different_length
            ],
            success_msg="embeddings and client ids have the same length",
        )

    def check_embeddings(
        self,
    ) -> List[str]:
        """
        Main method that checks embeddings and client_ids.

        Returns:
            List[str]: list containing reports for failed checks.
        """
        msg_if_client_ids_incorrect = self._err_msg_if_client_ids_incorrect()
        msg_if_embeddings_incorrect = self._err_msg_if_embeddings_incorrect()
        if msg_if_client_ids_incorrect is None and msg_if_embeddings_incorrect is None:
            msg_if_inconsistent = self._err_msg_if_inconsistent()
        else:
            msg_if_inconsistent = None
        potential_err_messages = [
            msg_if_client_ids_incorrect,
            msg_if_embeddings_incorrect,
            msg_if_inconsistent,
        ]
        err_messages = [
            err_msg for err_msg in potential_err_messages if err_msg is not None
        ]
        return err_messages
