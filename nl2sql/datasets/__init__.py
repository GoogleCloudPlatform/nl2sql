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
Implements utilities around Datasets and Databases
"""

from functools import lru_cache

from nl2sql.datasets.base import Dataset
from nl2sql.datasets.standard import Spider


@lru_cache
def fetch_dataset(dataset_id: str, **kwargs) -> Dataset:
    """
    Utility function to load standard datasets

    This function provides a convenient way to load standard datasets into memory.
    It currently supports the Spider dataset, but more datasets will be added in the future.

    Args:
        dataset_id: The ID of the dataset to fetch.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        Dataset: A Dataset object representing the requested dataset.
    """
    if dataset_id == "spider.train":
        return Spider().dataset(split="train", **kwargs)
    if dataset_id == "spider.test":
        return Spider().dataset(split="test", **kwargs)
    raise AttributeError(f"No known dataset found for {dataset_id}")
