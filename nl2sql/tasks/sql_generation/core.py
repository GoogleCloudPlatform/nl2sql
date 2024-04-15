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
Implementation of the core prompting based approach to SQL Generation
"""
from typing import Callable
from uuid import uuid4

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from langchain.llms.base import BaseLLM
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import BasePromptTemplate
from loguru import logger
from pydantic import BaseModel, SkipValidation
from typing_extensions import Literal

from nl2sql.assets.prompts import FewShot as FewShotPrompts
from nl2sql.assets.prompts import ZeroShot as ZeroShotPrompts
from nl2sql.datasets.base import Database
from nl2sql.tasks.sql_generation import BaseSqlGenerationResult, BaseSqlGenerationTask


class _CoreSqlGeneratorPrompt(BaseModel):
    """
    A Wrapper around SQL Generator Prompts
    """

    prompt_id: str
    dialect_prompt_template_map: dict[str, SkipValidation[BasePromptTemplate]]
    parser: SkipValidation[StructuredOutputParser] | None = None
    post_processor: Callable


class _SqlGeneratorPrompts:
    # pylint: disable=missing-function-docstring, invalid-name
    """
    Provides prompt options for generating SQL
    """

    default_parser: StructuredOutputParser = (
        StructuredOutputParser.from_response_schemas(
            [
                ResponseSchema(
                    name="thoughts",
                    description=(
                        "A short analysis of the question and available tables and "
                        "columns, demonstrating the thought process behind how the "
                        "query should be built."
                    ),
                ),
                ResponseSchema(
                    name="query",
                    description=(
                        "The correct SQL Query to answer the asked question. This "
                        "query should only contain information from above and not "
                        "use any external information."
                    ),
                ),
            ]
        )
    )

    @property
    def CURATED_ZERO_SHOT_PROMPT(self) -> _CoreSqlGeneratorPrompt:
        prompt_template = ZeroShotPrompts.TASK_SQL_GENERATION_CORE_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreSqlGeneratorPrompt(
            prompt_id="TASK_SQL_GENERATION_CORE_V1",
            dialect_prompt_template_map={"default": prompt_template},
            parser=self.default_parser,
            post_processor=lambda x: x.get("query"),
        )

    @property
    def CURATED_FEW_SHOT_COT_PROMPT(self) -> _CoreSqlGeneratorPrompt:
        prompt_template = FewShotPrompts.TASK_SQL_GENERATION_CORE_V1_SPIDER_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        prompt_template.example_prompt = prompt_template.example_prompt.partial(  # type: ignore
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreSqlGeneratorPrompt(
            prompt_id="TASK_SQL_GENERATION_CORE_V1_SPIDER_V1",
            dialect_prompt_template_map={"default": prompt_template},
            parser=self.default_parser,
            post_processor=lambda x: x.get("query"),
        )

    @property
    def LANGCHAIN_ZERO_SHOT_PROMPT(self) -> _CoreSqlGeneratorPrompt:
        return _CoreSqlGeneratorPrompt(
            prompt_id="LANGCHAIN_ZERO_SHOT_PROMPT",
            dialect_prompt_template_map={**SQL_PROMPTS, "default": PROMPT},
            parser=None,
            post_processor=lambda x: x.split("SQLResult:")[0]
            .split("SQLQuery:")[-1]
            .strip(),
        )

    @classmethod
    def custom_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        parser: StructuredOutputParser | None = None,
        post_processor: Callable = lambda x: x,
        prompt_template_id: str | None = None,
    ) -> _CoreSqlGeneratorPrompt:
        """
        Use a custom PromptTemplate for SQL Generation.
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

        return _CoreSqlGeneratorPrompt(
            prompt_id=f"CUSTOM-{prompt_template_id}",
            dialect_prompt_template_map={"default": prompt_template},
            post_processor=post_processor,
            parser=parser,
        )


prompts = _SqlGeneratorPrompts()


class CoreSqlGenratorResult(BaseSqlGenerationResult):
    """
    Implements Core SQL Generation Results
    """

    resulttype: Literal[
        "Result.SqlGeneration.CoreSqlGenerator"
    ] = "Result.SqlGeneration.CoreSqlGenerator"


class CoreSqlGenerator(BaseSqlGenerationTask):
    """
    Implements Core SQL Generation Task
    """

    tasktype: Literal[
        "Task.SqlGeneration.CoreSqlGenerator"
    ] = "Task.SqlGeneration.CoreSqlGenerator"

    llm: SkipValidation[BaseLLM]
    prompt: SkipValidation[_CoreSqlGeneratorPrompt] = prompts.LANGCHAIN_ZERO_SHOT_PROMPT

    def __call__(self, db: Database, question: str) -> CoreSqlGenratorResult:
        """
        Runs the SQL Generation pipeline
        """
        logger.info(f"Running {self.tasktype} ...")

        prompt_params = {
            "question": question,
            "query": question,
            "input": question,
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
            )
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
        intermediate_steps = [
            {
                "tasktype": self.tasktype,
                "prepared_prompt": prepared_prompt,
                "llm_response": llm_response.dict(),
                "raw_response": raw_response,
                "parsed_response": parsed_response,
                "processed_response": processed_response,
            }
        ]

        return CoreSqlGenratorResult(
            db_name=db.name,
            question=question,
            generated_query=processed_response,
            intermediate_steps=intermediate_steps,
        )
