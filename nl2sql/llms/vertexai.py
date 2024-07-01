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
from langchain_google_vertexai import VertexAI
from typing import ClassVar

class ExtendedVertexAI(VertexAI):
    """
    Adds utility functions to GooglePalm
    """
    model_token_limits: ClassVar[dict[str, int]] = {
        "text-bison" : {"input": 2048, "output": 1024},
        "text-bison-32k": {"input": 24000, "output": 8000},
        "gemini-1.5-flash-001": {"input": 1_040_000, "output": 8000},
        "gemini-1.0-pro-002": {"input": 24000, "output": 8000},
        "gemini-1.5-pro-001": {"input": 2_095_000, "output": 8000},
    }

    def get_num_tokens(self, text: str) -> int:
        """
        Returns the token count for some text
        """
        if (
            self.model_name.startswith("gemini")
            and
            self.model_name in self.model_token_limits
        ):
            return self.client.count_tokens(text).total_tokens
        else:
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
        if self.model_name not in self.model_token_limits:
            raise NotImplementedError
        else:
            return self.model_token_limits[self.model_name]["input"]
    
    def get_max_output_tokens(self) -> int:
        """
        Returns the maximum number of output tokens allowed
        """
        if self.model_name not in self.model_token_limits:
            raise NotImplementedError
        else:
            return self.model_token_limits[self.model_name]["output"]


def model(
        model_name="gemini-1.5-flash-001",
        max_output_tokens=8000,
        temperature=0.1,
        top_p=0.8,
        top_k=40) -> ExtendedVertexAI:
    """
    Return an Instance of Vertex AI LLM
    """
    if not model_name in ExtendedVertexAI.model_token_limits:
        raise NotImplementedError
    
    llm = ExtendedVertexAI(
        name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        n=1
    )
    model_output_tokens = llm.get_max_output_tokens()
    max_output_tokens =  min(model_output_tokens, max_output_tokens)
    llm.max_output_tokens = max_output_tokens
    return llm
    