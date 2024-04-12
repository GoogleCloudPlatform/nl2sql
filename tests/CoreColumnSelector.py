from nl2sql.datasets import fetch_dataset
from nl2sql.llms.vertexai import text_bison_latest
from nl2sql.tasks.column_selection.core import CoreColumnSelector, prompts

ds = fetch_dataset("spider.test")
db = ds.get_database("pets_1")
llm = text_bison_latest()

question = "Find the average weight for each pet type."

print("\n----------------------\nCTS 1\n----------------------\n")
ccs1 = CoreColumnSelector(llm=llm)
result1 = ccs1(db=db, question=question)
print(result1.intermediate_steps[0]["prepared_prompt"])
print(result1.intermediate_steps[0]["llm_response"]["generations"][0][0]["text"])

print("\n----------------------\nCTS 2\n----------------------\n")
ccs2 = CoreColumnSelector(llm=llm, prompt=prompts.CURATED_FEW_SHOT_COT_PROMPT)
result2 = ccs2(db=db, question=question)
print(result2.intermediate_steps[0]["prepared_prompt"])
print(result2.intermediate_steps[0]["llm_response"]["generations"][0][0]["text"])

print("done")
