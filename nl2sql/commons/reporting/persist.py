# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
