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
