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
Provides the base class for all Join Selection tasks
"""

from abc import ABC
from typing import Any

from nl2sql.datasets.base import Database
from nl2sql.tasks import BaseResult, BaseTask


class BaseJoinSelectionResult(BaseResult, ABC):
    """
    The core class for all Join Selection Task Results
    """

    resulttype: str = "Result.JoinSelection"
    db_name: str
    question: str
    allowed_joins: set[str]
    selected_joins: set[str]
    intermediate_steps: list[Any]


class BaseJoinSelectionTask(BaseTask, ABC):
    """
    The core class for all Join Selection Tasks
    """

    tasktype: str = "Task.JoinSelection"

    def __call__(self, db: Database, question: str) -> BaseJoinSelectionResult:
        raise NotImplementedError
