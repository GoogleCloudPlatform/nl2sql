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

"""
Provides the base class for all Executors
"""

import json
import os
from abc import ABC
from typing import Any
from uuid import uuid4

from langchain.llms.base import BaseLLM
from pydantic import BaseModel, Field, SkipValidation

from nl2sql.commons.reporting.persist import DEFAULT_HANDLER
from nl2sql.datasets import Dataset
from nl2sql.datasets.custom import CustomDataset


class BaseExecutor(BaseModel, ABC):
    """
    The core class for all Executors
    """

    executor_id: str = Field(default_factory=lambda: uuid4().hex)
    executortype: str = "Executor"
    dataset: Dataset

    @classmethod
    def from_connection_string_map(
        cls, connection_string_map: dict[str, str], **kwargs
    ):
        return cls(
            dataset=Dataset.from_connection_strings(
                name_connstr_map=connection_string_map, **kwargs
            ),
            **kwargs
        )

    @classmethod
    def from_excel(cls, filepath: str, dataset_name: str, project_id: str, **kwargs):
        return cls(
            dataset=CustomDataset.from_excel(
                project_id=project_id, filepath=filepath, dataset_name=dataset_name
            ),
            **kwargs
        )

    def model_post_init(self, __context: object) -> None:
        if os.environ.get("NL2SQL_ENABLE_ANALYTICS"):
            DEFAULT_HANDLER(
                artefact=json.loads(self.model_dump_json()),
                key=self.executortype,
                artefact_id=self.executor_id,
            )


class BaseResult(BaseModel, ABC):
    """
    The core class for all Executor Results
    """

    executor_id: str
    result_id: str = Field(default_factory=lambda: uuid4().hex)
    resulttype: str = "Result"
    intermediate_steps: list[Any]

    def model_post_init(self, __context: object) -> None:
        if os.environ.get("NL2SQL_ENABLE_ANALYTICS"):
            DEFAULT_HANDLER(
                artefact=json.loads(self.model_dump_json()),
                key=self.resulttype,
                artefact_id=self.result_id,
            )
