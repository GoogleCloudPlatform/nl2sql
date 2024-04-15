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
Implementation of the core prompting based approach to Join Selection
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
from nl2sql.tasks.join_selection import BaseJoinSelectionResult, BaseJoinSelectionTask


class _CoreJoinSelectorPrompt(BaseModel):
    """
    A Wrapper around Join Selector Prompts
    """

    prompt_id: str
    prompt_template: SkipValidation[BasePromptTemplate]
    parser: SkipValidation[StructuredOutputParser] | None = None
    post_processor: Callable


class _JoinSelectorPrompts:
    # pylint: disable=missing-function-docstring, invalid-name
    """
    Provides prompt options for selecting Joins before generating SQL
    """

    default_parser = StructuredOutputParser.from_response_schemas(
        [
            ResponseSchema(
                name="thoughts",
                description=(
                    "A short analysis of the question and available tables and "
                    "columns, demonstrating which joins would help in answering"
                    " the question and why. If no joins are needed, explain "
                    "why."
                ),
            ),
            ResponseSchema(
                name="joins",
                description=(
                    "A comma separated list of joining conditions that would "
                    "help in answering the question in the format "
                    "tablename1.columnname1=tablename2.columnname2. If no "
                    "joins are needed, this field should be null."
                ),
            ),
        ]
    )

    @property
    def CURATED_ZERO_SHOT_PROMPT(self) -> _CoreJoinSelectorPrompt:
        prompt_template = ZeroShotPrompts.TASK_JOIN_SELECTION_CORE_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreJoinSelectorPrompt(
            prompt_id="TASK_JOIN_SELECTION_CORE_V1",
            prompt_template=prompt_template,
            parser=self.default_parser,
            post_processor=lambda x: [
                i.strip().replace(" ", "") for i in x["joins"].split(",")
            ]
            if ((x) and (x.get("joins")))
            else [],
        )

    @property
    def CURATED_FEW_SHOT_COT_PROMPT(self) -> _CoreJoinSelectorPrompt:
        prompt_template = FewShotPrompts.TASK_JOIN_SELECTION_CORE_V1_SPIDER_V1.partial(
            format_instructions=self.default_parser.get_format_instructions()
        )
        prompt_template.example_prompt = prompt_template.example_prompt.partial(  # type: ignore
            format_instructions=self.default_parser.get_format_instructions()
        )
        return _CoreJoinSelectorPrompt(
            prompt_id="TASK_JOIN_SELECTION_CORE_V1_SPIDER_V1",
            prompt_template=prompt_template,
            parser=self.default_parser,
            post_processor=lambda x: [i.strip().replace(" ", "") for i in x["joins"]]
            if (x and isinstance(x.get("joins"), list))
            else [],
        )

    @classmethod
    def custom_prompt(
        cls,
        prompt_template: BasePromptTemplate,
        parser: StructuredOutputParser | None = None,
        post_processor: Callable = lambda x: x,
        prompt_template_id: str | None = None,
    ) -> _CoreJoinSelectorPrompt:
        """
        Use a custom PromptTemplate for Join filtering.
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

        return _CoreJoinSelectorPrompt(
            prompt_id=f"CUSTOM-{prompt_template_id}",
            prompt_template=prompt_template,
            post_processor=post_processor,
            parser=parser,
        )


prompts = _JoinSelectorPrompts()


class CoreJoinSelectorResult(BaseJoinSelectionResult):
    """
    Implements Core Join Selector Results
    """

    resulttype: Literal[
        "Result.JoinSelection.CoreJoinSelector"
    ] = "Result.JoinSelection.CoreJoinSelector"


class CoreJoinSelector(BaseJoinSelectionTask):
    """
    Implements Core Join Selector Task
    """

    tasktype: Literal[
        "Task.JoinSelection.CoreJoinSelector"
    ] = "Task.JoinSelection.CoreJoinSelector"

    llm: SkipValidation[BaseLLM]
    prompt: SkipValidation[_CoreJoinSelectorPrompt] = prompts.CURATED_ZERO_SHOT_PROMPT

    def __call__(self, db: Database, question: str) -> CoreJoinSelectorResult:
        """
        Runs the Join Selection pipeline
        """
        logger.info(f"Running {self.tasktype} ...")

        prompt_params = {
            "question": question,
            "query": question,
            "thoughts": [],
            "answer": None,
            "db_descriptor": {db.name: db.descriptor},
            "table_name": ", ".join(db.db._usable_tables),
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
        allowed_joins = {
            f"{tname}.{fk.parent.name}={getattr(fk, '_colspec')}"
            for tname, tobj in db.db._metadata.tables.items()
            for fk in tobj.foreign_keys
            if fk.parent is not None
        }

        allowed_joins_lower_map = {i.lower(): i for i in allowed_joins}

        selected_joins = {allowed_joins_lower_map.get(i, i) for i in processed_response}
        if not selected_joins:
            logger.critical("No Join Selected!")

        return CoreJoinSelectorResult(
            db_name=db.name,
            question=question,
            allowed_joins=allowed_joins,
            selected_joins=selected_joins,
            intermediate_steps=intermediate_steps,
        )
