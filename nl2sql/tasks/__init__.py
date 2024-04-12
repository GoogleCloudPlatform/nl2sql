"""
Provides the base class for all tasks
"""

from abc import ABC

from pydantic import BaseModel


class BaseTask(BaseModel, ABC):
    """
    The core class for all Tasks
    """

    tasktype: str = "Task"


class BaseResult(BaseModel, ABC):
    """
    The core class for all Task Results
    """

    resulttype: str = "Result"
