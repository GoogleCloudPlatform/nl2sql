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
Provides the base class for all Linear executors
"""

from abc import ABC

import pandas as pd

from nl2sql.datasets import Dataset
from nl2sql.executors import BaseExecutor, BaseResult


class BaseLinearExecutorResult(BaseResult, ABC):
    """
    The core class for all Linear Executor Results
    """

    resulttype: str = "Result.LinearExecutor"
    db_name: str
    question: str
    available_tables: set[str] | None
    selected_tables: set[str] | None
    available_columns: set[str] | None
    selected_columns: set[str] | None
    generated_query: str | None


class BaseLinearExecutor(BaseExecutor, ABC):
    """
    The core class for all Linear Executors
    """

    executortype: str = "Executor.LinearExecutor"

    def __call__(self, db_name: str, question: str) -> BaseLinearExecutorResult:
        raise NotImplementedError

    def fetch_result(self, result: BaseLinearExecutorResult) -> pd.DataFrame:
        if result.generated_query is None:
            raise ValueError("Supplied query is empty")
        return self.dataset.get_database(result.db_name).execute(result.generated_query)
