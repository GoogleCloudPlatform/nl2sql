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
