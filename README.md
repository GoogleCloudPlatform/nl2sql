# NL2SQL - A framework to convert natural language questions into SQL queries

## Introduction
NL2SQL is a library for building Natural Language to SQL workflows that are composable, explainable and extensible. 

- **Composability** : The NL2SQL library breaks down the process of translating a business question into a SQL query into smaller, atomic tasks, and provides specialised modules for each of these tasks, allowing you to create end-to-end NL2SQL flows that are fine-tuned and custom built for your data pipelines and your business requirements.

- **Explainability** : All of the tasks provide Chain-Of-Thoughts based options that allow you to gleam into how the LLM is interpreting the problem and strategising a solution. These "thoughts" not only allow post-hoc optimisations to prompts and parameters, but can also be exposed to the end user to help them draft their questions better.

- **Extensibility** : The tasks come with tested, well-performing default parameters, but also allow you to deeply customise them. Be it providing a new prompt template, a custom set of examples from your database, or a different LLM - each task is purpose built to accommodate diverse business needs. You can also build your own tasks and chain them with the rest of the workflow to extend your pipeline further.

