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
Used to get an instance of the PaLM LLM
"""
import os
from google.cloud import secretmanager
from langchain.llms.google_palm import GooglePalm


class ExtendedPalm(GooglePalm):
    """
    Adds utility functions to GooglePalm
    """

    def get_num_tokens(self, text: str):
        """
        Returns the token count for some text
        """
        return self.client.count_message_tokens(prompt=text)["token_count"]

    def get_max_input_tokens(self):
        """
        Returns the maximum number of input tokens allowed
        """
        return self.client.get_model("models/text-bison-001").input_token_limit


def get_secretmanager_authed_palm(
    project_id: str | None = None,
    secret_id: str = "palm-api-key",
    secret_version_id: str = "latest",
    **kwargs,
) -> ExtendedPalm:
    """
    Returns an Instance of ExtendedPalm already authed using the GSD API Key
    """
    if project_id is None:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    return ExtendedPalm(
        n=3,
        verbose=True,
        temperature=0.3,
        max_output_tokens=1024,
        **kwargs,
        google_api_key=secretmanager.SecretManagerServiceClient()
        .access_secret_version(
            name=f"projects/{project_id}/secrets/{secret_id}/versions/{secret_version_id}"
        )
        .payload.data.decode("UTF-8"),
    )
