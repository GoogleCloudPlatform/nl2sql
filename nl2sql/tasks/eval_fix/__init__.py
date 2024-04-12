"""
Provides the base class for all Eval and Fix tasks
"""

from abc import ABC
from typing import Any

from nl2sql.datasets.base import Database
from nl2sql.tasks import BaseResult, BaseTask


class BaseEvalFixResult(BaseResult, ABC):
    """
    The core class for all Eval Fix Task Results
    """
    resulttype: str = "Result.EvalFix"
    db_name: str
    question: str
    original_query: str
    modified_query: str
    intermediate_steps: list[Any]


class BaseEvalFixTask(BaseTask, ABC):
    """
    The core class for all Eval & Fix Tasks
    """
    tasktype: str = "Task.EvalFix"
    max_rows_limit: int = 1000

    def __call__(self,
                 db: Database,
                 question: str,
                 query: str) -> BaseEvalFixResult:
        raise NotImplementedError
