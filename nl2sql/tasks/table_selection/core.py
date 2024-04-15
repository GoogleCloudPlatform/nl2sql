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
Implementation of the core prompting based approach to Table Selection
"""
from typing import Callable
from uuid import uuid4

from langchain.chains.sql_database import prompt as lc_prompts
from langchain.llms.base import BaseLLM
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.schema import BasePromptTemplate
from loguru import logger
from pydantic import BaseModel, SkipValidation
from typing_extensions import Literal

from nl2sql.assets.prompts import FewShot as FewShotPrompts
from nl2sql.commons.utils.classifiers import yes_no_classifier
from nl2sql.datasets.base import Database
from nl2sql.tasks.table_selection import (
    BaseTableSelectionResult,
    BaseTableSelectionTask,
)


class _CoreTableSelectorPrompt(BaseModel):
    """
    A Wrapper around Table Selector Prompts to distinguish between iterative vs
    one-off table filtering prompts and allow simple output postprocessing
    """

    prompt_id: str
    prompt_template: SkipValidation[BasePromptTemplate]
    parser: StructuredOutputParser | None = None
    call_for_each_table: bool
    post_processor: Callable


class _TableSelectorPrompts:
    # pylint: disable=missing-function-docstring, invalid-name
    """
    Provides prompt options for selecting tables before generating SQL
    """

    @property
    def LANGCHAIN_DECIDER_PROMPT(self) -> _CoreTableSelectorPrompt:
        return _CoreTableSelectorPrompt(
            prompt_id="LANGCHAIN_DECIDER_PROMPT",
            prompt_template=lc_prompts.DECIDER_PROMPT,
            call_for_each_table=False,
            post_processor=lambda x: [i.strip() for i in x.split(",")],
        )

    @property
    def CURATED_FEW_SHOT_COT_PROMPT(self) -> _CoreTableSelectorPrompt:
        return _CoreTableSelectorPrompt(
            prompt_id="TASK_TABLE_SELECTION_CORE_V1_SPIDER_V1",
            prompt_template=FewShotPrompts.TASK_TABLE_SELECTION_CORE_V1_SPIDER_V1,
            call_for_each_table=True,
            post_processor=lambda x: yes_no_classifier(x) == "True",
        )

    @classmethod
    def custom_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        call_for_each_table: bool,
        parser: StructuredOutputParser | None = None,
        post_processor: Callable = lambda x: x,
        prompt_template_id: str | None = None,
    ) -> _CoreTableSelectorPrompt:
        """
        Use a custom PromptTemplate for table filtering.
        """
        if not prompt_template_id:
            prompt_template_id = uuid4().hex
        if parser and isinstance(prompt_template, FewShotPromptTemplate):
            prompt_template.example_prompt.partial_variables = {
                "format_instructions": parser.get_format_instructions()
            }

        return _CoreTableSelectorPrompt(
            prompt_id=f"CUSTOM-{prompt_template_id}",
            parser=parser,
            prompt_template=prompt_template,
            call_for_each_table=call_for_each_table,
            post_processor=post_processor,
        )


prompts = _TableSelectorPrompts()


class CoreTableSelectorResult(BaseTableSelectionResult):
    """
    Implements Core Table Selector Results
    """

    resulttype: Literal[
        "Result.TableSelection.CoreTableSelector"
    ] = "Result.TableSelection.CoreTableSelector"


class CoreTableSelector(BaseTableSelectionTask):
    """
    Implements Core Table Selector Task
    """

    tasktype: Literal[
        "Task.TableSelection.CoreTableSelector"
    ] = "Task.TableSelection.CoreTableSelector"

    llm: SkipValidation[BaseLLM]
    prompt: SkipValidation[_CoreTableSelectorPrompt] = prompts.LANGCHAIN_DECIDER_PROMPT

    def __call__(self, db: Database, question: str) -> CoreTableSelectorResult:
        """
        Runs the Table Selection pipeline
        """
        logger.info(f"Running {self.tasktype} ...")
        selected_tables = []
        intermediate_steps = []

        if self.prompt.call_for_each_table:
            targets = {
                tablename: {db.name: {tablename: tabledescriptor}}
                for tablename, tabledescriptor in db.descriptor.items()
            }
        else:
            targets = {",".join(db.db._usable_tables): {db.name: db.descriptor}}

        for tablename, dbdescriptor in targets.items():
            prompt_params = {
                "question": question,
                "query": question,
                "thoughts": [],
                "answer": None,
                "db_descriptor": dbdescriptor,
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
            if processed_response:
                if self.prompt.call_for_each_table:
                    selected_tables.append(tablename)
                else:
                    selected_tables = processed_response

        filtered_selected_tables: set[str] = set.intersection(
            set(selected_tables), db.db._usable_tables
        )
        if not filtered_selected_tables:
            logger.critical("No table Selected!")
        return CoreTableSelectorResult(
            db_name=db.name,
            question=question,
            available_tables=db.db._usable_tables,
            selected_tables=filtered_selected_tables,
            intermediate_steps=intermediate_steps,
        )
