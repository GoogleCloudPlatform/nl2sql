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
