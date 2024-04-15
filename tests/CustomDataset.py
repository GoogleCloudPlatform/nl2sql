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

from isort import file

from nl2sql.datasets import fetch_dataset
from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.tasks.column_selection.core import CoreColumnSelector
from nl2sql.tasks.column_selection.core import prompts as ccs_prompts
from nl2sql.tasks.sql_generation.core import CoreSqlGenerator
from nl2sql.tasks.sql_generation.core import prompts as csg_prompts
from nl2sql.tasks.table_selection.core import CoreTableSelector
from nl2sql.tasks.table_selection.core import prompts as cts_prompts

llm = text_bison_32k()

core_table_selector = CoreTableSelector(llm=llm, prompt=cts_prompts.CURATED_FEW_SHOT_COT_PROMPT)
core_column_selector = CoreColumnSelector(llm=llm, prompt=ccs_prompts.CURATED_FEW_SHOT_COT_PROMPT)
core_sql_generator = CoreSqlGenerator(llm=llm, prompt=csg_prompts.CURATED_FEW_SHOT_COT_PROMPT)

db_name="custom_dataset"
question = "What is the avg order price?"

print("\n----------------------\nCLE 1\n----------------------\n")
cle = CoreLinearExecutor.from_excel(
    filepath=".local/nl2sql_load_dataset.xlsx",
    dataset_name=db_name,
    project_id="gdc-ai-playground"
)
result = cle(db_name, question)
print(result.generated_query)

df = cle.fetch_result(result)
print(df)

print("\n----------------------\nCLE 2\n----------------------\n")
cle2 = CoreLinearExecutor.from_excel(
    filepath=".local/nl2sql_load_dataset.xlsx",
    dataset_name=db_name,
    project_id="gdc-ai-playground",
    core_table_selector = core_table_selector,
    core_column_selector = core_column_selector,
    core_sql_generator = core_sql_generator
)

result2 = cle2(db_name, question)
print(result2.generated_query)
print("Done")