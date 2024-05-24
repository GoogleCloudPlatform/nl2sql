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
from abc import ABC
from importlib.metadata import version
from pathlib import Path
from typing import Any

from google.cloud import storage  # type: ignore[attr-defined]
from loguru import logger

from nl2sql.commons.reporting.fingerprint import sys_info, user_info


class Persist(ABC):
    """
    Persists the artefacts
    """

    def __init__(self):
        try:
            self.user_info = user_info()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(str(exc))
            self.user_info = None
        try:
            self.sys_info = sys_info()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(str(exc))
            self.sys_info = None
        try:
            self.lib_version = version("nl2sql")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(str(exc))
            self.lib_version = "0.0.0"

    def get_data(self, artefact: dict[str, Any]):
        """
        Returns the data to be persisted
        """
        return json.dumps(
            {
                "data": artefact,
                "metadata": {
                    "system_info": self.sys_info,
                    "user_info": self.user_info,
                },
            }
        )

    def __call__(
        self,
        artefact_id: str,
        key: str,
        artefact: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class GCSPersist(Persist):
    """
    Persists the artefacts into GCS
    """

    def __init__(self, gcs_bucket: str):
        super().__init__()
        self.gcs_bucket = gcs_bucket

    def __call__(
        self,
        artefact_id: str,
        key: str,
        artefact: dict[str, Any],
    ) -> None:

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            storage.Client().get_bucket(LOG_BUCKET).blob(
                os.path.join(
                    "logs", key, self.lib_version, f"{artefact_id}_{timestamp}.json"
                )
            ).upload_from_string(
                data=self.get_data(artefact),
                content_type="application/json",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(str(exc))


class LocalPersist(Persist):
    """
    Persists the artefacts locally
    """

    def __call__(
        self,
        artefact_id: str,
        key: str,
        artefact: dict[str, Any],
    ) -> None:

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            output_file = Path(
                os.path.join("logs", key, f"{artefact_id}_{timestamp}.json")
            )
            output_file.parent.mkdir(exist_ok=True, parents=True)
            output_file.write_text(self.get_data(artefact))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(str(exc))


LOG_BUCKET = os.getenv("NL2SQL_LOG_BUCKET")

DEFAULT_HANDLER = GCSPersist(gcs_bucket=LOG_BUCKET) if LOG_BUCKET else LocalPersist()
