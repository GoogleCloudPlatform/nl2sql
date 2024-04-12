"""Allows persisting artefacts"""

import datetime
import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Any

from google.cloud import storage  # type: ignore[attr-defined]
from loguru import logger

from nl2sql.commons.reporting.fingerprint import sys_info, user_info

LOG_BUCKET = os.getenv("NL2SQL_LOG_BUCKET")
USER_INFO = user_info()
SYS_INFO = sys_info()


def gcs_handler(
    artefact_id: str,
    key: str,
    artefact: dict[str, Any],
) -> None:
    """
    Persists the artefacts into GCS
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        self_version = version("nl2sql")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        self_version = "0.0.0"
    try:
        data = json.dumps(
            {
                "data": artefact,
                "metadata": {"system_info": SYS_INFO, "user_info": USER_INFO},
            }
        )
        if LOG_BUCKET:
            storage.Client().get_bucket(LOG_BUCKET).blob(
                os.path.join(self_version, key, f"{artefact_id}_{timestamp}.json")
            ).upload_from_string(
                data=data,
                content_type="application/json",
            )
        else:

            output_file = Path(
                os.path.join("logs", key, f"{artefact_id}_{timestamp}.json")
            )
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file.write_text(data)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))


DEFAULT_HANDLER = gcs_handler
