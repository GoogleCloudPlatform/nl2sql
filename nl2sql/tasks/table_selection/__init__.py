"""
Provides the base class for all Table Selection tasks
"""

from abc import ABC
from typing import Any

from nl2sql.datasets.base import Database
from nl2sql.tasks import BaseResult, BaseTask


class BaseTableSelectionResult(BaseResult, ABC):
    """
    The core class for all Base Table Selection Task
    """

    resulttype: str = "Result.TableSelection"
    db_name: str
    question: str
    available_tables: set[str]
    selected_tables: set[str]
    intermediate_steps: list[Any]


class BaseTableSelectionTask(BaseTask, ABC):
    """
    The core class for all Base Table Selection Task Results
    """

    tasktype: str = "Task.TableSelection"

    def __call__(self, db: Database, question: str) -> BaseTableSelectionResult:
        raise NotImplementedError
