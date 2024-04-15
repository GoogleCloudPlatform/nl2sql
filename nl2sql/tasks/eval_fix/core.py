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
Implementation of the core prompting based approach to Eval and Fix for SQL.
"""
from typing import Callable, List
from uuid import uuid4

from langchain.llms.base import BaseLLM
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BasePromptTemplate
from loguru import logger
from pydantic import BaseModel, SkipValidation
from sqlalchemy.exc import DatabaseError
from tenacity import retry, stop_after_attempt
from typing_extensions import Literal

from nl2sql.assets.prompts import ZeroShot as ZeroShotPrompts
from nl2sql.datasets.base import Database
from nl2sql.tasks.eval_fix import BaseEvalFixResult, BaseEvalFixTask


class _CoreEvalFixPrompt(BaseModel):
    """
    A Wrapper around Eval & Fix Prompts
    """
    prompt_id: str
    dialect_prompt_template_map: dict[str, SkipValidation[BasePromptTemplate]]
    parser: SkipValidation[StructuredOutputParser] | None = None
    post_processor: Callable


class _EvalFixPrompts:
    # pylint: disable=missing-function-docstring, invalid-name
    """
    Provides prompt options for Eval & Fix of SQL
    """

    default_parser: StructuredOutputParser = (
            StructuredOutputParser.from_response_schemas(
                [
                    ResponseSchema(
                        name="thoughts",
                        description=(
                            "A short analysis of the question and available "
                            "tables and columns, demonstrating the thought "
                            "process behind how the query should be fixed."
                        ),
                    ),
                    ResponseSchema(
                        name="query",
                        description=(
                            "The syntactically correct and grounded SQL Query "
                            "to answer the asked question. This query should "
                            "only contain information from above and not "
                            "use any external information."
                        ),
                    ),
                ]
            )
        )

    @property
    def CURATED_ZERO_SHOT_PROMPT(self) -> _CoreEvalFixPrompt:
        prompt_template = ZeroShotPrompts.TASK_EVAL_FIX_CORE_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreEvalFixPrompt(
            prompt_id="TASK_EVAL_FIX_CORE_V1",
            dialect_prompt_template_map={"default": prompt_template},
            parser=self.default_parser,
            post_processor=lambda x: x.get("query"),
        )

    @classmethod
    def custom_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        parser: StructuredOutputParser | None = None,
        post_processor: Callable = lambda x: x,
        prompt_template_id: str | None = None,
    ) -> _CoreEvalFixPrompt:
        """
        Use a custom PromptTemplate for SQL Eval & Fix.
        """
        if not prompt_template_id:
            prompt_template_id = uuid4().hex
        if parser:
            prompt_template = prompt_template.partial(
                format_instructions=parser.get_format_instructions()
            )
            if hasattr(prompt_template, "example_prompt") and isinstance(
                getattr(prompt_template, "example_prompt"), PromptTemplate
            ):
                prompt_template.example_prompt = getattr(
                    prompt_template, "example_prompt"
                ).partial(format_instructions=parser.get_format_instructions())

        return _CoreEvalFixPrompt(
            prompt_id=f"CUSTOM-{prompt_template_id}",
            dialect_prompt_template_map={"default": prompt_template},
            post_processor=post_processor,
            parser=parser,
        )


prompts = _EvalFixPrompts()

class CoreEvalFixResult(BaseEvalFixResult):
    """
    Implements Core SQL Generation Results
    """

    resulttype: Literal[
        "Result.EvalFix.CoreEvalFix"
    ] = "Result.EvalFix.CoreEvalFix"


class CoreEvalFix(BaseEvalFixTask):
    """
    Implements Core Eval Fix Task.
    """

    tasktype: Literal[
        "Task.EvalFix.CoreEvalFix"
    ] = "Task.EvalFix.CoreEvalFix"

    llm: SkipValidation[BaseLLM]
    prompt: SkipValidation[_CoreEvalFixPrompt] = prompts.CURATED_ZERO_SHOT_PROMPT
    num_retries: int = 10

    def __call__(self,
                 db: Database,
                 question: str,
                 query: str
                ) -> CoreEvalFixResult:
        """
        Runs the Core Eval and Fix Pipeline

        Args:
            db (Database): Name of the database.
            question (str): Natural language query
            query (str): Generated SQL query that throws error.

        Returns:
            CoreEvalFixResult: Fixed Result.
        """
        logger.info(f"Running {self.tasktype} ...")
        original_query = query
        modified_query = query
        intermediate_steps = []
        trials: List[str]= []
        trials.append(modified_query)

        @retry(
                stop=stop_after_attempt(self.num_retries)
        )
        def evaluate():
            trial_id = len(trials)
            sql = trials[-1]
            logger.info(f"Trial Id: {trial_id}")
            logger.info(f"Evaluating Generated Query: {sql}")
            try:
                _ = db.execute(sql) # type: ignore
            except DatabaseError as db_error:
                error_message = db_error.args[0].splitlines()[0]
                logger.warning(f"Evaluation Failed: "
                             f"{error_message}")
                logger.debug("Trying to fix the query ...")
                prompt_params = {
                    "question": question,
                    "query": question,
                    "input": question,
                    "generated_query": trials[-1],
                    "error_message": error_message,
                    "thoughts": [],
                    "answer": None,
                    "dialect": db.db.dialect,
                    "top_k": self.max_rows_limit,
                    "table_info": db.db.table_info,
                    "db_descriptor": {db.name: db.descriptor},
                    "table_name": ", ".join(db.db._usable_tables),
                    "table_names": list(db.db._usable_tables),
                }

                prompt_template = self.prompt.dialect_prompt_template_map.get(
                        db.db.dialect,
                        self.prompt.dialect_prompt_template_map.get("default"),
                    )
                if prompt_template is None:
                    raise ValueError(
                        f"No suitable / default prompt template found for {db.db.dialect}"
                    ) from db_error
                prepared_prompt = prompt_template.format(
                    **{
                        k: v
                        for k, v in prompt_params.items()
                        if k in prompt_template.input_variables
                    }
                )

                llm_response = self.llm.generate([prepared_prompt])
                logger.debug(
                    f"[{self.tasktype}] : Received LLM Response : {llm_response.json()}"
                )
                try:
                    raw_response = llm_response.generations[0][0].text.strip()
                except IndexError as exc:
                    raise ValueError(
                        f"Empty / Invalid Response received from LLM : {llm_response.json()}"
                    ) from exc

                parsed_response = (
                    self.prompt.parser.parse(raw_response)
                    if self.prompt.parser
                    else raw_response
                )
                processed_response = self.prompt.post_processor(parsed_response)
                intermediate_steps.append(
                    {
                        f"trial_{trial_id}": {
                                    "tasktype": self.tasktype,
                                    "prepared_prompt": prepared_prompt,
                                    "llm_response": llm_response.dict(),
                                    "raw_response": raw_response,
                                    "parsed_response": parsed_response,
                                    "processed_response": processed_response,
                                }
                    }
                )
                logger.info(f"New generated query: {processed_response}")
                trials.append(processed_response)
                trial_id += 1
                raise RuntimeError("Retry Evaluation...") from db_error
            else:
                logger.success("Generated Query successfully evaluated ...")
                return sql

        # Eval and Fix
        try:
            output =  evaluate()
            logger.success("EvalFix Successful.")
        except Exception as exc:
            logger.error(f"EvalFix Failed: {exc}")
            output = trials[-1]

        evalfixresult = CoreEvalFixResult(
            db_name=db.name,
            question=question,
            original_query=original_query,
            modified_query=output,
            intermediate_steps=intermediate_steps
        )
        logger.debug(
                    f"[{evalfixresult.resulttype}] : Final Query: {output}"
                )
        return evalfixresult
