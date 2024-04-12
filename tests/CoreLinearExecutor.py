from nl2sql.datasets import fetch_dataset
from nl2sql.executors.linear_executor.core import CoreLinearExecutor
from nl2sql.llms.vertexai import text_bison_latest

ds = fetch_dataset("spider.test")
db_name="pets_1"
question = "Find the average weight for each pet type."

print("\n----------------------\CLE 1\n----------------------\n")
cle = CoreLinearExecutor(dataset=ds)
result = cle(db_name, question)
print("Done")