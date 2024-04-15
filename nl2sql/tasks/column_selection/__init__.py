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
Provides the base class for all Column Selection tasks
"""

from abc import ABC
from typing import Any

from nl2sql.datasets.base import Database
from nl2sql.tasks import BaseResult, BaseTask


class BaseColumnSelectionResult(BaseResult, ABC):
    """
    The core class for all Column Selection Task Results
    """

    resulttype: str = "Result.ColumnSelection"
    db_name: str
    question: str
    available_columns: set[str]
    selected_columns: set[str]
    intermediate_steps: list[Any]


class BaseColumnSelectionTask(BaseTask, ABC):
    """
    The core class for all Column Selection Tasks
    """

    tasktype: str = "Task.ColumnSelection"

    def __call__(self, db: Database, question: str) -> BaseColumnSelectionResult:
        raise NotImplementedError
