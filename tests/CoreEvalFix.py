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

from nl2sql.datasets import Dataset
from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.tasks.eval_fix.core import CoreEvalFix

db = Dataset.from_connection_strings(
    name_connstr_map={
        "libraries_io": "bigquery://<YOUR-PROJECT-HERE>/libraries_io",
    }
)

question = "What is the name of the project with the highest source rank?"
incorrect_query = "SELECT my_name FROM projects ORDER BY sourcerank DESC LIMIT 1"
eval_fix_task = CoreEvalFix(llm=text_bison_32k(), num_retries=10)
eval_fix_task(db.databases["libraries_io"], question, incorrect_query)

print("done")
