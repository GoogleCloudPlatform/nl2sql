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
