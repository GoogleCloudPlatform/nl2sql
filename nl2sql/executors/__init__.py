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
