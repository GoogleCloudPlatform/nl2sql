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

from nl2sql.datasets import fetch_dataset
from nl2sql.llms.vertexai import text_bison_latest
from nl2sql.tasks.join_selection.core import CoreJoinSelector, prompts

ds = fetch_dataset("spider.train")
db = ds.get_database("department_management")
llm = text_bison_latest()

question = "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"

print("\n----------------------\nCTS 1\n----------------------\n")
cjs1 = CoreJoinSelector(llm=llm)
result1 = cjs1(db=db, question=question)
print(result1.intermediate_steps[0]["prepared_prompt"])
print(result1.intermediate_steps[0]["llm_response"]["generations"][0][0]["text"])

print("\n----------------------\nCTS 2\n----------------------\n")
cjs2 = CoreJoinSelector(llm=llm, prompt=prompts.CURATED_FEW_SHOT_COT_PROMPT)
result2 = cjs2(db=db, question=question)
print(result2.intermediate_steps[0]["prepared_prompt"])
print(result2.intermediate_steps[0]["llm_response"]["generations"][0][0]["text"])

print("done")
