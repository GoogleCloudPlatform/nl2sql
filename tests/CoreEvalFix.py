from nl2sql.datasets import Dataset
from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.tasks.eval_fix.core import CoreEvalFix

db = Dataset.from_connection_strings(
            name_connstr_map = {
                "libraries_io": "bigquery://gdc-ai-playground/libraries_io",
            }
        )

question = "What is the name of the project with the highest source rank?"
incorrect_query = "SELECT my_name FROM projects ORDER BY sourcerank DESC LIMIT 1"
eval_fix_task = CoreEvalFix(llm=text_bison_32k(), num_retries=10)
eval_fix_task(db.databases["libraries_io"], question, incorrect_query)

print("done")
