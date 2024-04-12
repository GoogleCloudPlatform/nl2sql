from nl2sql.datasets import fetch_dataset
from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from nl2sql.llms.vertexai import text_bison_32k
from nl2sql.tasks.eval_fix.core import CoreEvalFix

ds = fetch_dataset("spider.test")
db_name="pets_1"
question = "Find the average weight for each pet type."
eval_fix_task = CoreEvalFix(llm=text_bison_32k(), num_retries=10)
print("\n----------------------\CLE 1\n----------------------\n")
cle = CoreLinearExecutor(
                            dataset=ds,
                            core_eval_fix=eval_fix_task
                        )
result = cle(db_name, question)
for step in result.intermediate_steps:
    print(result.intermediate_steps,"\n\n")
print("Done")