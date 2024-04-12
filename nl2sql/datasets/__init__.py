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
