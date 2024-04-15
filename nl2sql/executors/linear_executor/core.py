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
Implementation of the linear core executor
"""
import re

from loguru import logger
from pydantic import Field
from typing_extensions import Literal

from nl2sql.executors.linear_executor import (
    BaseLinearExecutor,
    BaseLinearExecutorResult,
)
from nl2sql.llms.vertexai import text_bison_32k, text_bison_latest
from nl2sql.tasks.column_selection import BaseColumnSelectionTask
from nl2sql.tasks.column_selection.core import CoreColumnSelector
from nl2sql.tasks.eval_fix import BaseEvalFixTask
from nl2sql.tasks.eval_fix.core import CoreEvalFix
from nl2sql.tasks.sql_generation import BaseSqlGenerationTask
from nl2sql.tasks.sql_generation.core import CoreSqlGenerator
from nl2sql.tasks.table_selection import BaseTableSelectionTask
from nl2sql.tasks.table_selection.core import CoreTableSelector


class CoreLinearExecutorResult(BaseLinearExecutorResult):
    """
    Implements Core Linear Executor Results
    """

    resulttype: Literal[
        "Result.LinearExecutor.CoreLinearExecutor"
    ] = "Result.LinearExecutor.CoreLinearExecutor"


class CoreLinearExecutor(BaseLinearExecutor):
    """
    Implements Core Linear Executor
    """

    executortype: Literal[
        "Executor.LinearExecutor.CoreLinearExecutor"
    ] = "Executor.LinearExecutor.CoreLinearExecutor"

    core_table_selector: BaseTableSelectionTask | None = Field(
        default_factory=lambda: CoreTableSelector(llm=text_bison_32k())
    )
    core_column_selector: BaseColumnSelectionTask | None = Field(
        default_factory=lambda: CoreColumnSelector(llm=text_bison_32k())
    )
    core_sql_generator: BaseSqlGenerationTask = Field(
        default_factory=lambda: CoreSqlGenerator(llm=text_bison_32k())
    )
    core_eval_fix: BaseEvalFixTask = Field(
        default_factory=lambda: CoreEvalFix(llm=text_bison_32k())
    )


    def __call__(self, db_name: str, question: str) -> CoreLinearExecutorResult:
        """
        Runs the Core Linear Executor
        """
        logger.info(f"Running {self.executortype} ...")
        database = self.dataset.get_database(db_name)
        result_intermediate_steps = []
        if self.core_table_selector is not None:
            result_ts = self.core_table_selector(db=database, question=question)
            database = database.filter(
                filters=[f"{db_name}.{i}.*" for i in result_ts.selected_tables],
                filter_type="only",
            )
            result_available_tables = result_ts.available_tables
            result_selected_tables = result_ts.selected_tables
            result_intermediate_steps.append(
                {"table_selection": result_ts.intermediate_steps}
            )
        else:
            result_available_tables = None
            result_selected_tables = None

        if self.core_column_selector is not None:
            result_cs = self.core_column_selector(db=database, question=question)
            database = database.filter(
                filters=[f"{db_name}.{i}" for i in result_cs.selected_columns],
                filter_type="only",
            )
            result_available_columns = result_cs.available_columns
            result_selected_columns = result_cs.selected_columns
            result_intermediate_steps.append(
                {"column_selection": result_cs.intermediate_steps}
            )
        else:
            result_available_columns = None
            result_selected_columns = None

        result_sg = self.core_sql_generator(db=database, question=question)
        result_generated_query = result_sg.generated_query

        result_intermediate_steps.append(
            {"sql_generation": result_sg.intermediate_steps}
        )

        if self.core_eval_fix is not None:
            if result_generated_query:
                try:
                    eval_fix_result = self.core_eval_fix(
                        db=database,
                        question=question,
                        query=result_generated_query
                    )
                except Exception as exc:
                    logger.error(f"EvalFix failed: {exc}")
                else:
                    result_intermediate_steps.append(
                        {"eval_fix": eval_fix_result.intermediate_steps}
                    )
                    result_generated_query = eval_fix_result.modified_query
        
        #Generated SQL cleanup : Remove Backticks if any
        if result_generated_query is not None:
            result_generated_query = re.sub("```|sql", "",
                                            result_generated_query)

        return CoreLinearExecutorResult(
            db_name=db_name,
            question=question,
            executor_id=self.executor_id,
            available_tables=result_available_tables,
            selected_tables=result_selected_tables,
            available_columns=result_available_columns,
            selected_columns=result_selected_columns,
            generated_query=result_generated_query,
            intermediate_steps=result_intermediate_steps,
        )
