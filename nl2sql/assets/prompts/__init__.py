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
provides prompts as PromptTemplate objects
"""

import json
import pkgutil
from typing import List

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from loguru import logger
from pydantic.v1 import BaseModel

from nl2sql.assets.examples import FewShot as FewShotExamples


class _ZeroShot:
    # pylint: disable=invalid-name, missing-function-docstring
    def _create_template(self, filename: str) -> PromptTemplate:
        raw_data = pkgutil.get_data(__name__, filename)
        if raw_data is None:
            raise ValueError(f"{filename} cannot be read")
        template = json.loads(raw_data)
        return PromptTemplate.from_template(
            template="".join(template["template"]),
            template_format=template["template_format"],
        )

    @property
    def COLUMN_DESCRIPTION_V1(self) -> PromptTemplate:
        return self._create_template("column_description_v1.json")

    @property
    def COLUMN_DESCRIPTION_V2(self) -> PromptTemplate:
        return self._create_template("column_description_v2.json")

    @property
    def TABLE_DESCRIPTION_V1(self) -> PromptTemplate:
        return self._create_template("table_description_v1.json")

    @property
    def TABLE_DESCRIPTION_V2(self) -> PromptTemplate:
        return self._create_template("table_description_v2.json")

    @property
    def TABLE_DESCRIPTION_V3(self) -> PromptTemplate:
        return self._create_template("table_description_v3.json")

    @property
    def TABLE_FILTER_THOUGHT_GEN_V1(self) -> PromptTemplate:
        return self._create_template("table_filter_thought_gen_v1.json")

    @property
    def TABLE_FILTER_THOUGHT_SCORE_V1(self) -> PromptTemplate:
        return self._create_template("table_filter_thought_score_v1.json")

    @property
    def TABLE_FILTER_THOUGHT_SCORE_V2(self) -> PromptTemplate:
        return self._create_template("table_filter_thought_score_v2.json")

    @property
    def COLUMN_FILTER_THOUGHT_GEN_V1(self) -> PromptTemplate:
        return self._create_template("column_filter_thought_gen_v1.json")

    @property
    def COLUMN_FILTER_THOUGHT_SCORE_V1(self) -> PromptTemplate:
        return self._create_template("column_filter_thought_score_v1.json")

    @property
    def PROMPTING_STRAT_QUERY_RANK_V1(self) -> PromptTemplate:
        return self._create_template("prompting_strat_query_rank_v1.json")

    @property
    def PROMPTING_STRAT_SQL_GEN_V1(self) -> PromptTemplate:
        return self._create_template("prompting_strat_sql_gen_v1.json")

    @property
    def PROMPTING_STRAT_TABLE_FILTER_GEN_V1(self) -> PromptTemplate:
        return self._create_template("prompting_strat_table_filter_gen_v1.json")

    @property
    def PROMPTING_STRAT_COLUMN_FILTER_GEN_V1(self) -> PromptTemplate:
        return self._create_template("prompting_strat_column_filter_gen_v1.json")

    @property
    def TASK_TABLE_SELECTION_CORE_V1(self) -> PromptTemplate:
        return self._create_template("task_table_selection_core_v1.json")

    @property
    def TASK_COLUMN_SELECTION_CORE_V1(self) -> PromptTemplate:
        return self._create_template("task_column_selection_core_v1.json")

    @property
    def TASK_JOIN_SELECTION_CORE_V1(self) -> PromptTemplate:
        return self._create_template("task_join_selection_core_v1.json")

    @property
    def TASK_SQL_GENERATION_CORE_V1(self) -> PromptTemplate:
        return self._create_template("task_sql_generation_core_v1.json")
    
    @property
    def TASK_EVAL_FIX_CORE_V1(self) -> PromptTemplate:
        return self._create_template("task_eval_fix_core_v1.json")


ZeroShot = _ZeroShot()


class _FewShot:
    # pylint: disable=invalid-name, missing-function-docstring
    def _create_template_v1(
        self,
        example_prompt: PromptTemplate,
        input_variables: List[str],
        examples: List[dict],
        num_examples: int,
    ) -> FewShotPromptTemplate:
        return FewShotPromptTemplate(
            example_prompt=example_prompt,
            input_variables=input_variables,
            suffix=example_prompt.template,
            examples=examples[:num_examples],
            template_format=example_prompt.template_format,
        )

    def _create_template_v2(
        self,
        example_prompt: PromptTemplate,
        examples: List[dict],
        num_examples: int,
    ) -> FewShotPromptTemplate:
        from nl2sql.datasets import (  # pylint: disable=import-outside-toplevel
            fetch_dataset,
        )

        # Importing this globally would make the entire prompt module dependent
        # on Datasets, while Datasets already depends on prompts, resulting in
        # a cyclic import error. Importing this here makes only _FewShotPrompts
        # dependednt on Datasets, while dataset does not depend on _FewShotPrompts

        dataset_map = {j: fetch_dataset(j) for j in {i["dataset"] for i in examples}}
        extended_examples = []
        for e in examples:
            db_descriptor = {
                db.name: db.descriptor
                for db in (
                    dataset_map[e["dataset"]]
                    .filter(filters=e["data_id"], filter_type="only")
                    .databases.values()
                )
                if db.descriptor
            }
            if db_descriptor:
                extended_examples.append(
                    {
                        **e,
                        "db_descriptor": db_descriptor,
                    }
                )

            if len(extended_examples) >= num_examples:
                break
        else:
            raise ValueError(f"Unable to load {num_examples} Examples")

        return FewShotPromptTemplate(
            example_prompt=example_prompt,
            input_variables=example_prompt.input_variables,
            suffix=example_prompt.template,
            examples=extended_examples,
            template_format=example_prompt.template_format,
        )

    @property
    def PROMPTING_STRAT_FEW_SHOT_SQL_GEN_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating PROMPTING_STRAT_FEW_SHOT_SQL_GEN_V1")
        return self._create_template_v1(
            ZeroShot.PROMPTING_STRAT_SQL_GEN_V1,
            input_variables=["context", "question", "query"],
            examples=FewShotExamples.EXAMPLES_SPIDER_SQL_QUERIES_V1,
            num_examples=5,
        )

    @property
    def PROMPTING_STRAT_FEW_SHOT_TABLE_FILTER_GEN_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating PROMPTING_STRAT_FEW_SHOT_TABLE_FILTER_GEN_V1")
        return self._create_template_v1(
            ZeroShot.PROMPTING_STRAT_TABLE_FILTER_GEN_V1,
            input_variables=["context", "question", "thoughts"],
            examples=FewShotExamples.EXAMPLES_TABLE_FILTER_V1,
            num_examples=5,
        )

    @property
    def PROMPTING_STRAT_FEW_SHOT_COLUMN_FILTER_GEN_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating PROMPTING_STRAT_FEW_SHOT_COLUMN_FILTER_GEN_V1")
        return self._create_template_v1(
            ZeroShot.PROMPTING_STRAT_COLUMN_FILTER_GEN_V1,
            input_variables=["context", "question", "thoughts", "answer"],
            examples=FewShotExamples.EXAMPLES_COLUMN_FILTER_V1,
            num_examples=5,
        )

    @property
    def SPIDER_FEW_SHOT_COLUMN_FILTER_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating SPIDER_FEW_SHOT_COLUMN_FILTER_V1")
        return self._create_template_v2(
            ZeroShot.PROMPTING_STRAT_COLUMN_FILTER_GEN_V1,
            examples=FewShotExamples.EXAMPLES_COLUMN_FILTER_V2,
            num_examples=5,
        )

    @property
    def TASK_TABLE_SELECTION_CORE_V1_SPIDER_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating TASK_TABLE_SELECTION_CORE_V1_SPIDER_V1")
        return self._create_template_v2(
            ZeroShot.TASK_TABLE_SELECTION_CORE_V1,
            examples=FewShotExamples.EXAMPLES_TABLE_FILTER_V2,
            num_examples=5,
        )

    @property
    def TASK_COLUMN_SELECTION_CORE_V1_SPIDER_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating TASK_COLUMN_SELECTION_CORE_V1_SPIDER_V1")
        return self._create_template_v2(
            ZeroShot.TASK_COLUMN_SELECTION_CORE_V1,
            examples=FewShotExamples.EXAMPLES_COLUMN_FILTER_V2,
            num_examples=5,
        )

    @property
    def TASK_JOIN_SELECTION_CORE_V1_SPIDER_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating TASK_JOIN_SELECTION_CORE_V1_SPIDER_V1")
        return self._create_template_v2(
            ZeroShot.TASK_JOIN_SELECTION_CORE_V1,
            examples=FewShotExamples.EXAMPLES_JOIN_IDENTIFICATION_V2,
            num_examples=5,
        )

    @property
    def TASK_SQL_GENERATION_CORE_V1_SPIDER_V1(self) -> FewShotPromptTemplate:
        logger.info("Instantiating TASK_SQL_GENERATION_CORE_V1_SPIDER_V1")
        return self._create_template_v2(
            ZeroShot.TASK_SQL_GENERATION_CORE_V1,
            examples=FewShotExamples.EXAMPLES_SQL_GENERATION_V1,
            num_examples=2,
        )


FewShot = _FewShot()
