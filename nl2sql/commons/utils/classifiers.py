"""
Allows classifying unstructured text into defined categories.
"""

from typing import Literal
from langchain.embeddings import VertexAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

_yes_no_classifier = Chroma.from_documents(
    [
        Document(page_content="No", metadata={"relevant": "False"}),
        Document(page_content="Yes", metadata={"relevant": "True"}),
    ],
    VertexAIEmbeddings(),
).as_retriever(search_type="similarity", search_kwargs={"k": 1})


def yes_no_classifier(target: str) -> Literal["True", "False"]:
    """
    Categorizes arbitrary incoming string into "True"/"False" lterals
    """
    return _yes_no_classifier.get_relevant_documents(target.lower())[0].metadata[
        "relevant"
    ]
