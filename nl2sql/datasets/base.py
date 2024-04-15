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
Implements core Dataset functionality.
A dataset is a group of databases, intended to represent a data warehouse
containing multiple databases.
"""

import re
import typing

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    SkipValidation,
    field_serializer,
    field_validator,
)
from pydantic.networks import UrlConstraints
from pydantic_core import Url
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.sql import expression as sqe
from sqlalchemy.sql.ddl import CreateTable
from sqlalchemy.sql.functions import func
from sqlalchemy.sql.schema import MetaData
from sqlalchemy.sql.sqltypes import VARCHAR
from typing_extensions import Self, TypedDict

from nl2sql.assets.prompts import ZeroShot as ZeroShotPrompts

ColName = typing.TypeVar("ColName", bound=str)
ColType = typing.TypeVar("ColType", bound=str)
ColDesc = typing.TypeVar("ColDesc", bound=str)
TabName = typing.TypeVar("TabName", bound=str)
TabDesc = typing.TypeVar("TabDesc", bound=str)
DBName = typing.TypeVar("DBName", bound=str)
DBDesc = typing.TypeVar("DBDesc", bound=str)
EnumValues = dict[DBName, dict[ColName, list[str]]]
ColDataDictionary = TypedDict(
    "ColDataDictionary", {"description": ColDesc, "type": ColType}
)
BaseColDataDictionary = dict[ColName, ColDataDictionary]
TabDataDictionary = TypedDict(
    "TabDataDictionary",
    {"description": TabDesc, "columns": BaseColDataDictionary},
)
BaseTabDataDictionary = dict[TabName, TabDataDictionary]
DatabaseDataDictionary = TypedDict(
    "DatabaseDataDictionary",
    {"description": DBDesc, "tables": BaseTabDataDictionary},
)
DatasetDataDictionary = dict[DBName, DatabaseDataDictionary]
BaseTableSchema = dict[ColName, ColType]
BaseDatabaseSchema = dict[TabName, BaseTableSchema]
BaseDatasetSchema = dict[DBName, BaseDatabaseSchema]
DatabaseSchema = TypedDict(
    "DatabaseSchema", {"metadata": MetaData, "tables": BaseDatabaseSchema}
)
DatasetSchema = dict[DBName, DatabaseSchema]
BaseColDescriptor = TypedDict(
    "BaseColDescriptor",
    {
        "col_type": str,
        "col_nullable": bool,
        "col_pk": bool,
        "col_defval": typing.Any | None,
        "col_comment": str | None,
        "col_enum_vals": list[str] | None,
        "col_description": str | None,
    },
)
BaseTabDescriptor = TypedDict(
    "BaseTabDescriptor",
    {
        "table_name": TabName,
        "table_creation_statement": str,
        "table_sample_rows": str,
        "col_descriptor": dict[str, BaseColDescriptor],
    },
)
TableDescriptor = dict[TabName, BaseTabDescriptor]

AllowedDSN = typing.Annotated[
    Url,
    UrlConstraints(
        host_required=False,
        allowed_schemes=["postgres", "postgresql", "mysql", "bigquery", "sqlite"],
    ),
]


class EntitySet(BaseModel):
    """
    Expects a list of  identifiers in the form of databasename.tablename.columnname
    """

    ids: list[str]
    dataset_schema: BaseDatasetSchema

    @field_validator("ids")
    @classmethod
    def id_structure(cls, ids: list[str]) -> list[str]:
        """
        Validates incoming IDs
        """
        for curr_id in ids:
            assert curr_id != "*.*.*", '"*.*.*" is not allowed'
            assert (
                len(id_parts := curr_id.split(".")) == 3
            ), f"Malformed Entity ID {curr_id}"
            for id_part, part_type in zip(id_parts, ["database", "table", "column"]):
                assert id_part == "*" or re.match(
                    "^[a-zA-Z0-9_-]+$", id_part
                ), f"Malformed {part_type} '{id_part}' in {curr_id}"
        return ids

    def __hash__(self) -> int:
        """
        Provides hash for the object
        """
        return ",".join(sorted(self.ids)).__hash__()

    def filter(
        self, key: typing.Literal["database", "table", "column"], value: str
    ) -> "EntitySet":
        """
        Generates a new entityset by filtering the current dataset
        based on the key and values passed
        """
        new_ids = []
        for curr_id in self.ids:
            db_name, tab_name, col_name = curr_id.split(".")
            if {"database": db_name, "table": tab_name, "column": col_name}[
                key
            ] == value:
                new_ids.append(curr_id)
        return EntitySet(
            ids=new_ids,
            dataset_schema=self.dataset_schema,
        )

    def invert(self) -> "EntitySet":
        """
        Returnes the complement of the provided keys based on the schema.
        """
        return EntitySet(
            ids=list(
                {
                    f"{db}.{tab}.{col}"
                    for db, tabval in self.dataset_schema.items()
                    for tab, colval in tabval.items()
                    for col in colval.keys()
                }
                - set(self.ids)
            ),
            dataset_schema=self.dataset_schema,
        )

    def prune_schema(self) -> BaseDatasetSchema:
        """
        Reduces the schema to only contain the keys present in the provided IDs
        """
        schema: BaseDatasetSchema = {}
        for curr_id in self.ids:
            dbname, tabname, colname = curr_id.split(".")
            if dbname not in schema:
                schema[dbname] = {}
            if tabname not in schema[dbname]:
                schema[dbname][tabname] = {}
            if colname not in schema[dbname][tabname]:
                schema[dbname][tabname][colname] = self.dataset_schema[dbname][tabname][
                    colname
                ]
        return schema

    def model_post_init(self, __context: object) -> None:
        stack = list(set(self.ids))
        resolved_ids: list[str] = []
        while stack:
            curr_id = stack.pop()
            database, table, column = curr_id.split(".")
            if database == "*":
                stack.extend(
                    [f"{db}.{table}.{column}" for db in self.dataset_schema.keys()]
                )
            elif table == "*":
                stack.extend(
                    [
                        f"{database}.{tab}.{column}"
                        for tab in self.dataset_schema.get(database, {}).keys()
                    ]
                )
            elif column == "*":
                stack.extend(
                    [
                        f"{database}.{table}.{col}"
                        for col in self.dataset_schema.get(database, {})
                        .get(table, {})
                        .keys()
                    ]
                )
            elif (
                (database in self.dataset_schema.keys())
                and (table in self.dataset_schema[database].keys())
                and (column in self.dataset_schema[database][table].keys())
            ):
                resolved_ids.append(curr_id)
            else:
                logger.debug(f"Invalid Filter Expression Found: {curr_id}. Skipping.")
        self.ids = resolved_ids


class Database(BaseModel):
    """
    Implements the core Database class which provides various utilities for a DB
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    db: SQLDatabase
    dsn: AllowedDSN
    dbschema: BaseDatabaseSchema
    enum_limit: int = 10
    descriptor: TableDescriptor = {}
    exclude_entities: EntitySet = EntitySet(ids=[], dataset_schema={})
    data_dictionary: DatabaseDataDictionary | None = None
    table_desc_template: SkipValidation[PromptTemplate] = (
        ZeroShotPrompts.TABLE_DESCRIPTION_V3
    )
    # TODO @madhups - find and remove all uses of SkipValidation across all
    # modules after Langchain has been ported to use Pydantic V2:
    # https://github.com/langchain-ai/langchain/discussions/9337

    @field_serializer("table_desc_template")
    def serialize_prompt_template(self, table_desc_template: PromptTemplate, _info):
        """
        Langchain Serializer
        """
        return {
            "template": table_desc_template.template,
            "template_format": table_desc_template.template_format,
        }

    @field_serializer("db")
    def serialize_db_basic(self, db: SQLDatabase, _info):
        # pylint: disable=protected-access
        """
        Langchain Serializer
        """
        return {
            "engine.url": db._engine.url.render_as_string(),
            "_all_tables": db._all_tables,
            "_usable_tables": db._usable_tables,
            "_sample_rows_in_table_info": db._sample_rows_in_table_info,
            "_indexes_in_table_info": db._indexes_in_table_info,
            "_custom_table_info": db._custom_table_info,
            "_max_string_length": db._max_string_length,
        }

    @classmethod
    def fetch_schema(cls, name: str, dsn: AllowedDSN) -> BaseDatasetSchema:
        """
        Queries the dtabase to find out the schema for a given db name and DSN
        """
        logger.info(f"[{name}] : Fetching Schema ...")
        metadata = MetaData()
        metadata.reflect(
            bind=create_engine(dsn.unicode_string()),
            views=True,
        )
        db_schema: BaseDatasetSchema = {name: {}}
        if metadata.tables:
            for tablename, table in metadata.tables.items():
                tabledata = {}
                for column in table.columns:
                    tabledata[column.name] = str(column.type)
                if not tabledata:
                    msg = f"No columns found in {name}.{tablename}"
                    logger.critical(msg)
                    raise AttributeError(msg)
                db_schema[name][tablename] = tabledata
        if not db_schema[name]:
            msg = f"No tables found in {name}"
            logger.critical(msg)
            raise AttributeError(msg)
        logger.success(f"[{name}] : Schema Obtained Successfully")
        return db_schema

    @classmethod
    def from_connection_string(
        cls,
        name: str,
        connection_string: str,
        schema: BaseDatasetSchema | None = None,
        **kwargs,
    ) -> Self:
        """
        Utility function to create a database from a name and connection_string
        """
        logger.debug(f"[{name}] : Analysing ...")
        dsn = AllowedDSN(connection_string)
        schema = schema or cls.fetch_schema(name=name, dsn=dsn)
        engine = create_engine(connection_string)
        assert isinstance(engine, Engine)
        if ("exclude_entities" in kwargs) and (
            not isinstance(kwargs["exclude_entities"], EntitySet)
        ):
            kwargs["exclude_entities"] = [
                EntitySet(ids=list(kwargs["exclude_entities"]), dataset_schema=schema)
            ]

        db = SQLDatabase(
            engine=engine,
            view_support=True,
        )
        logger.success(f"[{name}] : Analysis Complete")
        return cls(
            name=name,
            dsn=dsn,
            dbschema=schema[name],
            db=db,
            **kwargs,
        )

    def filter(
        self, filters: list[str], filter_type: typing.Literal["only", "exclude"]
    ) -> "Database":
        """
        Returns a new database object after applying the provided filters
        """
        entities = EntitySet(ids=filters, dataset_schema={self.name: self.dbschema})
        if filter_type == "only":
            entities = entities.invert()
        return Database(
            name=self.name,
            db=self.db,
            dsn=self.dsn,
            dbschema=self.dbschema,
            enum_limit=self.enum_limit,
            descriptor=self.descriptor,
            exclude_entities=entities,
            data_dictionary=self.data_dictionary,
            table_desc_template=self.table_desc_template,
        )

    def execute(self, query: str) -> pd.DataFrame:
        """
        Returns the results of a query as a Pandas DataFrame
        """
        return pd.read_sql(sql=query, con=self.db._engine)

    def model_post_init(self, __context: object) -> None:
        # pylint: disable=protected-access, too-many-branches
        """
        Langchain's Post-Init method to properly validate DB
        """
        logger.debug(f"[{self.name}] : Instantiating ...")
        logger.debug(f"[{self.name}] : Calculating Exclusions ...")
        table_exclusions = []
        all_exclusions = set(self.exclude_entities.ids)
        for tablename, tableinfo in self.dbschema.items():
            if {
                f"{self.name}.{tablename}.{column}" for column in tableinfo.keys()
            }.issubset(all_exclusions):
                table_exclusions.append(tablename)
        if table_exclusions:
            logger.info(
                f"[{self.name}] : These tables will be excluded :"
                + (", ".join(table_exclusions))
            )
        else:
            logger.info(f"[{self.name}] : No tables will be excluded")
        logger.success(f"[{self.name}] : Exclusions Calculated")
        logger.debug(f"[{self.name}] : Generating Custom Descriptions ...")
        engine = create_engine(self.dsn.unicode_string())
        assert isinstance(engine, Engine)
        temp_db = SQLDatabase(
            engine=engine,
            ignore_tables=table_exclusions,
            view_support=True,
        )
        table_descriptor: dict[str, BaseTabDescriptor] = {}
        table_descriptions = {}
        for table in temp_db._metadata.sorted_tables:
            if table.name in table_exclusions:
                continue
            table_descriptor[table.name] = {
                "table_name": table.name,
                "table_creation_statement": str(
                    CreateTable(table).compile(engine)
                ).rstrip(),
                "table_sample_rows": temp_db._get_sample_rows(table),
                "col_descriptor": {},
            }
            constraints = {
                col.name
                for con in table.constraints
                for col in con.columns  # type: ignore
            }
            col_enums = []
            for col in table._columns:  # type: ignore
                if (col.name not in constraints) and (
                    f"{self.name}.{table.name}.{col.name}" in all_exclusions
                ):
                    logger.info(
                        f"[{self.name}.{table.name}] : Removing column {col.name}"
                    )
                    table._columns.remove(col)  # type: ignore
                else:
                    if (table.name not in self.descriptor) or (
                        col.name not in self.descriptor[table.name]["col_descriptor"]
                    ):
                        if (self.enum_limit > 0) and (col.type.python_type == str):
                            col_enums.append(
                                sqe.select(
                                    sqe.literal(col.name, VARCHAR).label("COLNAME"),
                                    sqe.case(
                                        (
                                            sqe.select(
                                                func.count(sqe.distinct(col))
                                                < self.enum_limit
                                            ).label("COLCOUNT"),
                                            col,
                                        )
                                    ).label("COLVALS"),
                                ).distinct()
                            )

                        col_descriptor: BaseColDescriptor = {
                            "col_type": str(col.type),
                            "col_nullable": col.nullable,
                            "col_pk": col.primary_key,
                            "col_defval": col.default,
                            "col_comment": col.comment,
                            "col_enum_vals": None,
                            "col_description": (
                                (
                                    self.data_dictionary["tables"][table.name][
                                        "columns"
                                    ][col.name]["description"]
                                )
                                if (
                                    (self.data_dictionary)
                                    and (table.name in self.data_dictionary["tables"])
                                    and (
                                        col.name
                                        in self.data_dictionary["tables"][table.name][
                                            "columns"
                                        ]
                                    )
                                )
                                else None
                            ),
                        }
                    else:
                        col_descriptor = self.descriptor[table.name]["col_descriptor"][
                            col.name
                        ]
                    table_descriptor[table.name]["col_descriptor"][
                        col.name
                    ] = col_descriptor

            for colname, colvals in (
                (
                    pd.read_sql(sql=sqe.union(*col_enums), con=engine)
                    .replace("", np.nan)
                    .dropna()
                    .groupby("COLNAME", group_keys=False)["COLVALS"]
                    .apply(list)
                    .to_dict()
                )
                if col_enums
                else {}
            ).items():
                table_descriptor[table.name]["col_descriptor"][colname][
                    "col_enum_vals"
                ] = colvals
            table_descriptions[table.name] = self.table_desc_template.format(
                **{
                    key: value
                    for key, value in table_descriptor[table.name].items()
                    if key in self.table_desc_template.input_variables
                }
            )

        self.descriptor = table_descriptor
        logger.success(f"[{self.name}] : Custom Descriptions Generated")
        temp_db._custom_table_info = table_descriptions
        self.db = temp_db
        logger.success(f"[{self.name}] : Instantiated")


DBNameDBMap = dict[DBName, Database]


class Dataset(BaseModel):
    """
    A dataset is a collection of databases
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    databases: DBNameDBMap
    dataset_schema: BaseDatasetSchema
    exclude_entities: EntitySet = EntitySet(ids=[], dataset_schema={})
    data_dictionary: DatasetDataDictionary = {}
    enum_limit: int = 10
    table_desc_template: SkipValidation[PromptTemplate] = (
        ZeroShotPrompts.TABLE_DESCRIPTION_V3
    )
    # TODO @madhups - remove SkipValidation after Pydantic V2 support is added
    # to Langchain: https://github.com/langchain-ai/langchain/discussions/9337

    def get_database(self, database_name: str) -> Database:
        """
        Utility function to fetch a specific database from a dataset
        """
        return self.databases[database_name]

    def filter(
        self,
        filters: list[str],
        filter_type: typing.Literal["only", "exclude"],
        prune: bool = False,
    ) -> "Dataset":
        """
        Applies a filter to this dtaset and provides a new instance.
        """
        databases = {
            k: v.filter(filters, filter_type) for k, v in self.databases.items()
        }
        if prune:
            databases = {k: v for k, v in databases if v.db.table_info}
        return Dataset(
            databases=databases,
            dataset_schema=self.dataset_schema,
            exclude_entities=self.exclude_entities,
            data_dictionary=self.data_dictionary,
            enum_limit=self.enum_limit,
            table_desc_template=self.table_desc_template,
        )

    @property
    def list_databases(self) -> list[str]:
        """
        Returns a list of databases in this dataset
        """
        return list(self.databases.keys())

    @field_serializer("table_desc_template")
    def serialize_prompt_template(self, table_desc_template: PromptTemplate, _info):
        """
        Langchain Serializer
        """
        return {
            "template": table_desc_template.template,
            "template_format": table_desc_template.template_format,
        }

    @classmethod
    def from_connection_strings(
        cls,
        name_connstr_map: dict[str, str],
        exclude_entities: list[str] = [],
        **kwargs,
    ) -> Self:
        """
        Utility function to create a dataset from a name -> conn_str mapping.
        """
        dataset_schema = {
            db_name: Database.fetch_schema(db_name, AllowedDSN(db_connstr))[db_name]
            for db_name, db_connstr in name_connstr_map.items()
        }
        parsed_exclude_entities = EntitySet(
            ids=exclude_entities, dataset_schema=dataset_schema
        )

        data_dictionary = kwargs.pop("data_dictionary", dict())
        databases = {
            db_name: Database.from_connection_string(
                name=db_name,
                connection_string=db_connstr,
                exclude_entities=parsed_exclude_entities.filter(
                    key="database", value=db_name
                ),
                schema={db_name: dataset_schema[db_name]},
                data_dictionary=data_dictionary.get(db_name),
                **kwargs,
            )
            for db_name, db_connstr in name_connstr_map.items()
        }
        return cls(
            databases=databases,
            dataset_schema=dataset_schema,
            exclude_entities=parsed_exclude_entities,
            data_dictionary=data_dictionary,
            **kwargs,
        )
