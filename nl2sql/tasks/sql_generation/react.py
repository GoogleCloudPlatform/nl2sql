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
Implementation of the ReAct prompting based approach to SQL Generation
"""
from typing import Any

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.llms.base import BaseLLM
from loguru import logger
from pydantic import SkipValidation
from typing_extensions import Literal

from nl2sql.datasets.base import Database
from nl2sql.tasks.sql_generation import BaseSqlGenerationResult, BaseSqlGenerationTask


class ReactSqlGenratorResult(BaseSqlGenerationResult):
    """
    Implements ReAct SQL Generation Results
    """

    resulttype: Literal[
        "Result.SqlGeneration.ReactSqlGenerator"
    ] = "Result.SqlGeneration.ReactSqlGenerator"


class ReactSqlGenerator(BaseSqlGenerationTask):
    """
    Implements ReAct SQL Generation Task
    """

    tasktype: Literal[
        "Task.SqlGeneration.ReactSqlGenerator"
    ] = "Task.SqlGeneration.ReactSqlGenerator"

    llm: SkipValidation[BaseLLM]
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    max_iterations: int | None = 15

    def __call__(self, db: Database, question: str) -> ReactSqlGenratorResult:
        """
        Runs the SQL Generation pipeline
        """
        intermediate_steps: list[Any] = []
        logger.info(f"Running {self.tasktype} ...")
        agent = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=db.db, llm=self.llm),
            agent_type=self.agent_type,
            top_k=self.max_rows_limit,
            max_iterations=self.max_iterations,
            verbose=False,
            early_stopping_method="generate",
        )
        agent.return_intermediate_steps = True
        agent.handle_parsing_errors = True
        try:
            result = agent(question)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            intermediate_steps.append(f"Exception in Agent.run : {exc}")
            query = None
        else:
            isteps = result.get("intermediate_steps")
            intermediate_steps.extend(
                [
                    {"input": step[0].to_json(), "output": step[1]}
                    if isinstance(step, tuple)
                    else step
                    for step in isteps
                ]
                if isteps
                else []
            )

            try:
                query = next(
                    map(
                        lambda x: x[0]
                        .tool_input.replace(";", "")
                        .replace("sql```", "")
                        .replace("```sql", "")
                        .replace("```", ""),
                        filter(
                            lambda x: x[0].tool
                            in ["sql_db_query", "sql_db_query_checker"],
                            reversed(result.get("intermediate_steps", [])),
                        ),
                    ),
                    None,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                query = None
                intermediate_steps.append(f"Exception in parsing query : {exc}")

        return ReactSqlGenratorResult(
            db_name=db.name,
            question=question,
            generated_query=query,
            intermediate_steps=intermediate_steps,
        )
