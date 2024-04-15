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
Implementation of the core prompting based approach to Column Selection
"""
from typing import Callable
from uuid import uuid4

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
from nl2sql.tasks.column_selection import (
    BaseColumnSelectionResult,
    BaseColumnSelectionTask,
)


class _CoreColumnSelectorPrompt(BaseModel):
    """
    A Wrapper around Column Selector Prompts
    """

    prompt_id: str
    prompt_template: SkipValidation[BasePromptTemplate]
    parser: SkipValidation[StructuredOutputParser] | None = None
    post_processor: Callable


class _ColumnSelectorPrompts:
    # pylint: disable=missing-function-docstring, invalid-name
    """
    Provides prompt options for selecting Columns before generating SQL
    """

    default_parser = StructuredOutputParser.from_response_schemas(
        [
            ResponseSchema(
                name="thoughts",
                description=(
                    "A single sentence analysis of each column's relevance "
                    "to the question asked, followed by a Yes / No indicating "
                    "whether the column is relevant to the question."
                ),
            ),
            ResponseSchema(
                name="columns",
                description=(
                    "A comma separated list of column names relevant to "
                    "the question in the format tablename.columnname"
                ),
            ),
        ]
    )

    @property
    def CURATED_ZERO_SHOT_PROMPT(self) -> _CoreColumnSelectorPrompt:
        prompt_template = ZeroShotPrompts.TASK_COLUMN_SELECTION_CORE_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreColumnSelectorPrompt(
            prompt_id="TASK_COLUMN_SELECTION_CORE_V1",
            prompt_template=prompt_template,
            parser=self.default_parser,
            post_processor=lambda x: [i.strip() for i in x["columns"].split(",")]
            if ((x) and (x.get("columns")))
            else [],
        )

    @property
    def CURATED_FEW_SHOT_COT_PROMPT(self) -> _CoreColumnSelectorPrompt:
        prompt_template = (
            FewShotPrompts.TASK_COLUMN_SELECTION_CORE_V1_SPIDER_V1.partial(
                format_instructions=self.default_parser.get_format_instructions()
            )
        )
        prompt_template.example_prompt = prompt_template.example_prompt.partial(  # type: ignore
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreColumnSelectorPrompt(
            prompt_id="TASK_COLUMN_SELECTION_CORE_V1_SPIDER_V1",
            prompt_template=prompt_template,
            parser=self.default_parser,
            post_processor=lambda x: [i.strip() for i in x["columns"]]
            if (x and isinstance(x.get("columns"), list))
            else [],
        )

    @classmethod
    def custom_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        parser: StructuredOutputParser | None = None,
        post_processor: Callable = lambda x: x,
        prompt_template_id: str | None = None,
    ) -> _CoreColumnSelectorPrompt:
        """
        Use a custom PromptTemplate for Column filtering.
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
        return _CoreColumnSelectorPrompt(
            prompt_id=f"CUSTOM-{prompt_template_id}",
            prompt_template=prompt_template,
            post_processor=post_processor,
            parser=parser,
        )


prompts = _ColumnSelectorPrompts()


class CoreColumnSelectorResult(BaseColumnSelectionResult):
    """
    Implements Core Column Selector Results
    """

    resulttype: Literal[
        "Result.ColumnSelection.CoreColumnSelector"
    ] = "Result.ColumnSelection.CoreColumnSelector"


class CoreColumnSelector(BaseColumnSelectionTask):
    """
    Implements Core Column Selector Task
    """

    tasktype: Literal[
        "Task.ColumnSelection.CoreColumnSelector"
    ] = "Task.ColumnSelection.CoreColumnSelector"

    llm: SkipValidation[BaseLLM]
    prompt: SkipValidation[_CoreColumnSelectorPrompt] = prompts.CURATED_ZERO_SHOT_PROMPT

    def __call__(self, db: Database, question: str) -> CoreColumnSelectorResult:
        """
        Runs the Column Selection pipeline
        """
        logger.info(f"Running {self.tasktype} ...")
        selected_columns = []
        intermediate_steps = []

        for tablename, tabledescriptor in db.descriptor.items():
            prompt_params = {
                "question": question,
                "query": question,
                "thoughts": [],
                "answer": None,
                "db_descriptor": {db.name: {tablename: tabledescriptor}},
                "table_name": tablename,
                "table_names": list(db.db._usable_tables),
            }
            prepared_prompt = self.prompt.prompt_template.format(
                **{
                    k: v
                    for k, v in prompt_params.items()
                    if k in self.prompt.prompt_template.input_variables
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
                    "tasktype": self.tasktype,
                    "table": tablename,
                    "prepared_prompt": prepared_prompt,
                    "llm_response": llm_response.dict(),
                    "raw_response": raw_response,
                    "parsed_response": parsed_response,
                    "processed_response": processed_response,
                }
            )
            selected_columns.extend(processed_response)

        avialble_columns = {
            f"{tabname}.{colname}"
            for tabname, tabdesc in db.descriptor.items()
            for colname in tabdesc["col_descriptor"].keys()
        }
        avialble_columns_lower_map = {i.lower(): i for i in avialble_columns}
        filtered_selected_columns: set[str] = {
            avialble_columns_lower_map[c.lower()]
            for c in selected_columns
            if c.lower() in avialble_columns_lower_map
        }
        if not filtered_selected_columns:
            logger.critical("No column Selected!")
        return CoreColumnSelectorResult(
            db_name=db.name,
            question=question,
            available_columns=avialble_columns,
            selected_columns=filtered_selected_columns,
            intermediate_steps=intermediate_steps,
        )
