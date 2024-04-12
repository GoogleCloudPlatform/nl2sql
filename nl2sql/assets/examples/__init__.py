"""
Provides Few shot examples as dict objects
"""

import json
import pkgutil
import typing


class _FewShot:
    # pylint: disable=invalid-name, missing-function-docstring
    def _load(self, filename: str) -> list[dict[str, typing.Any]]:
        data = pkgutil.get_data(__name__, filename)
        if data is None:
            raise AttributeError(f"Could not read {filename}")
        return json.loads(data)

    @property
    def EXAMPLES_COLUMN_FILTER_V1(self) -> list[dict[str, typing.Any]]:
        return self._load("column_filter_v1.json")

    @property
    def EXAMPLES_TABLE_FILTER_V1(self) -> list[dict[str, typing.Any]]:
        return self._load("table_filter_v1.json")

    @property
    def EXAMPLES_JOIN_IDENTIFICATION_V1(self) -> list[dict[str, typing.Any]]:
        return self._load("join_identification_v1.json")

    @property
    def EXAMPLES_COLUMN_FILTER_V2(self) -> list[dict[str, typing.Any]]:
        return self._load("column_filter_v2.json")

    @property
    def EXAMPLES_TABLE_FILTER_V2(self) -> list[dict[str, typing.Any]]:
        return self._load("table_filter_v2.json")

    @property
    def EXAMPLES_JOIN_IDENTIFICATION_V2(self) -> list[dict[str, typing.Any]]:
        return self._load("join_identification_v2.json")

    @property
    def EXAMPLES_SQL_GENERATION_V1(self) -> list[dict[str, typing.Any]]:
        return self._load("sql_generation_v1.json")

    @property
    def EXAMPLES_SPIDER_SQL_QUERIES_V1(self) -> list[dict[str, typing.Any]]:
        return self._load("spider_sql_queries_v1.json")


FewShot = _FewShot()
