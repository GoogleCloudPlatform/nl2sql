from nl2sql.datasets import fetch_dataset
from nl2sql.llms.vertexai import text_bison_latest
from nl2sql.tasks.sql_generation.react import ReactSqlGenerator

ds = fetch_dataset("spider.test")
db = ds.get_database("pets_1")
llm = text_bison_latest()

question = "Find the average weight for each pet type."

print("\n----------------------\nCTS 1\n----------------------\n")
rsg1 = ReactSqlGenerator(llm=llm)
result1 = rsg1(db=db, question=question)

print("done")
