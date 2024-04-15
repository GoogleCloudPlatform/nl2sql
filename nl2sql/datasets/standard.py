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
Allows importing and Spider Dataset
"""
import json
import os
import typing
from abc import ABC
from tempfile import gettempdir
from zipfile import ZipFile

import datasets as hf_dataset
from datasets import config as hf_config
from google.cloud import storage  # type: ignore[attr-defined]
from typing_extensions import TypedDict

from nl2sql.datasets.base import Dataset


class StandardDataset(ABC):
    """
    Base class for all Standard Datasets
    """

    dataset_id: str = "StandardDataset"
    dataset_splits: list[str] = []

    def __init__(self) -> None:
        """
        Method to auomatically download and
        set up the dataset
        """
        raise NotImplementedError

    def dataset(self, **kwargs) -> Dataset:
        """
        Method to return subbsets of the
        Standard Dataset object as Dataset objects
        """
        raise NotImplementedError


SpiderCoreSpec = TypedDict(
    "SpiderCoreSpec",
    {
        "db_id": str,
        "query": str,
        "conn_str": str,
        "query_toks": list[str],
        "query_toks_no_value": list[str],
        "question": str,
        "question_toks": list[str],
        "sql": dict[
            typing.Literal[
                "from",
                "select",
                "where",
                "groupBy",
                "having",
                "orderBy",
                "limit",
                "intersect",
                "union",
                "except",
            ],
            typing.Any,
        ],
    },
)


class Spider(StandardDataset):
    dataset_id: str = "Spider"
    dataset_splits: list[str] = ["test", "train"]
    bucket_name: str = "nl2sql-internal"
    zipfile_path: str = "assets/datasets/spider/spider.zip"
    temp_loc = os.path.join(gettempdir(), "NL2SQL_SPIDER_DATASET")
    temp_extracted_loc = os.path.join(
        gettempdir(), "NL2SQL_SPIDER_DATASET", "extracted"
    )
    promblematic_databases: typing.ClassVar[
        dict[typing.Literal["errors", "warnings"], list[str]]
    ] = {
        "errors": [
            "wta_1",
            "soccer_1",
            "baseball_1",
            "store_1",
            "flight_1",
            "sakila_1",
            "world_1",
            "store_product",
            "college_1",
            "music_1",
            "loan_1",
            "hospital_1",
            "tracking_grants_for_research",  # Special Characters in column name
            "aircraft",  # Special Characters in column name
            "perpetrator",  # Special Characters in column name
            "orchestra",  # Special Characters in column name
        ],
        "warnings": [
            "bike_1",
            "cre_Drama_Workshop_Groups",
            "apartment_rentals",
            "insurance_and_eClaims",
            "soccer_2",
            "tracking_grants_for_research",
            "customer_deliveries",
            "dog_kennels",
            "chinook_1",
            "real_estate_properties",
            "department_store",
            "twitter_1",
            "products_for_hire",
            "manufactory_1",
            "college_2",
            "tracking_share_transactions",
            "hr_1",
            "customers_and_invoices",
            "customer_complaints",
            "behavior_monitoring",
            "aircraft",
            "solvency_ii",
        ],
    }

    def __init__(self) -> None:
        """
        Method to auomatically download and
        set up the Spider dataset
        """
        if not (
            os.path.exists(os.path.join(self.temp_extracted_loc,
                                        "spider",
                                        "train_spider.json"))
            or
            os.path.exists(os.path.join(self.temp_extracted_loc,
                                        "spider",
                                        "dev.json"))
            or
            os.path.exists(os.path.join(self.temp_extracted_loc,
                                        "spider",
                                        "database"))
        ):
            if not os.path.exists(self.temp_extracted_loc):
                os.makedirs(self.temp_extracted_loc)
            temp_zipfile_path = os.path.join(self.temp_loc, "spider.zip")
            if not os.path.exists(temp_zipfile_path):
                storage.Client().get_bucket(self.bucket_name).blob(
                        self.zipfile_path
                    ).download_to_filename(temp_zipfile_path)  
            with ZipFile(temp_zipfile_path, "r") as zipped_file:
                zipped_file.extractall(path=self.temp_extracted_loc)

    def fetch_raw_data(
        self, split: typing.Literal["test", "train"], strict: bool = False
    ) -> list[SpiderCoreSpec]:
        """
        Returns the raw data from the Spider Dataset
        """
        base_loc = os.path.join(self.temp_extracted_loc, "spider")
        database_loc = os.path.join(base_loc, "database")
        split_file_loc = os.path.join(
            base_loc, {"test": "dev.json", "train": "train_spider.json"}[split]
        )
        with open(split_file_loc, encoding="utf-8") as split_file:
            raw_data = [
                typing.cast(
                    SpiderCoreSpec,
                    {
                        **i,
                        "conn_str": f"sqlite:///{database_loc}/{i['db_id']}/{i['db_id']}.sqlite",
                    },
                )
                for i in json.load(split_file)
                if (i["db_id"] not in self.promblematic_databases.get("errors", []))
                and (
                    (not strict)
                    or (
                        i["db_id"]
                        not in self.promblematic_databases.get("warnings", [])
                    )
                )
            ]
        return raw_data

    def dataset(self, **kwargs) -> Dataset:
        """
        Creates and returns a Dataset object based on the specified Spider split
        """
        return Dataset.from_connection_strings(
            name_connstr_map=dict(
                {
                    (i["db_id"], i["conn_str"])
                    for i in self.fetch_raw_data(
                        split=kwargs["split"], strict=kwargs.get("strict", False)
                    )
                }
            ),
            **kwargs,
        )
