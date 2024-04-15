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
Used to get an instance of the Vertex AI LLM
"""
from google.cloud import aiplatform_v1beta1
from google.protobuf import struct_pb2
from langchain.llms.vertexai import VertexAI


class ExtendedVertexAI(VertexAI):
    """
    Adds utility functions to GooglePalm
    """

    def get_num_tokens(self, text: str) -> int:
        """
        Returns the token count for some text
        """
        token_struct = struct_pb2.Struct()
        token_struct.update({"content": text})
        return (
            aiplatform_v1beta1.PredictionServiceClient(
                client_options={
                    "api_endpoint": f"{self.location}-aiplatform.googleapis.com"
                }
            )
            .count_tokens(
                endpoint=self.client._endpoint_name,  # pylint: disable = protected-access
                instances=[struct_pb2.Value(struct_value=token_struct)],
            )
            .total_tokens
        )

    def get_max_input_tokens(self) -> int:
        """
        Returns the maximum number of input tokens allowed
        """
        if self.metadata:
            return self.metadata["max_input_tokens"]
        raise ValueError("LLM initialized without max_input_tokens")


def text_bison_latest(
    max_output_tokens=1024, temperature=0.1, top_p=0.8, top_k=40, candidate_count=3
) -> ExtendedVertexAI:
    """
    Returns an Instance of Vertex AI LLM
    """
    return ExtendedVertexAI(
        model_name="text-bison",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=candidate_count,
        metadata={"max_input_tokens": 3000},
    )


def text_bison_32k(
    max_output_tokens=8000, temperature=0.1, top_p=0.8, top_k=40
) -> ExtendedVertexAI:
    """
    Returns an Instance of Vertex AI LLM
    """
    return ExtendedVertexAI(
        model_name="text-bison-32k",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1,
        metadata={"max_input_tokens": 24000},
    )
