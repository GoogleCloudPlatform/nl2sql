"""
Provides the base class for all SQL Generation tasks
"""

from abc import ABC
from typing import Any

from nl2sql.datasets.base import Database
from nl2sql.tasks import BaseResult, BaseTask


class BaseSqlGenerationResult(BaseResult, ABC):
    """
    The core class for all SQL Generation Task Results
    """

    resulttype: str = "Result.SqlGeneration"
    db_name: str
    question: str
    generated_query: str | None
    intermediate_steps: list[Any]


class BaseSqlGenerationTask(BaseTask, ABC):
    """
    The core class for all SQL Generation Tasks
    """

    tasktype: str = "Task.SqlGeneration"
    max_rows_limit: int = 1000

    def __call__(self, db: Database, question: str) -> BaseSqlGenerationResult:
        raise NotImplementedError
