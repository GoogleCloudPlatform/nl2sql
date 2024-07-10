import unittest
from unittest.mock import MagicMock
from loguru import logger
from nl2sql.datasets.base import Database
from nl2sql.tasks.table_selection.core import (
    CoreTableSelector,
    _TableSelectorPrompts,
)

class TestCoreTableSelector(unittest.TestCase):
    def test_call_with_langchain_decider_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            generations=[
                [MagicMock(text="TableA, TableB")]
            ]
        )
        selector = CoreTableSelector(
            llm=mock_llm, prompt=_TableSelectorPrompts().LANGCHAIN_DECIDER_PROMPT
        )
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.db._usable_tables = {"TableA", "TableC"}
        mock_db.descriptor = {
            "TableA": "Description A",
            "TableB": "Description B",
            "TableC": "Description C",
        }
        result = selector(mock_db, "test question")
        self.assertEqual(result.selected_tables, {"TableA"})
        self.assertEqual(result.db_name, "test_db")
        self.assertEqual(result.question, "test question")
        self.assertEqual(result.available_tables, {"TableA", "TableC"})

    def test_call_with_curated_few_shot_cot_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            MagicMock(
                generations=[
                    [MagicMock(text="Yes. TableA is relevant")]
                ]
            ),
            MagicMock(
                generations=[
                    [MagicMock(text="No. TableB is not relevant")]
                ]
            ),
        ]
        selector = CoreTableSelector(
            llm=mock_llm, prompt=_TableSelectorPrompts().CURATED_FEW_SHOT_COT_PROMPT
        )
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.db._usable_tables = {"TableA", "TableB"}
        mock_db.descriptor = {
            "TableA": {
                "col_descriptor": {
                    "column1": "data_type",
                    "column2": "data_type"
                }
            },
            "TableB": {
                "col_descriptor": {
                    "column1": "data_type",
                    "column2": "data_type"
                }
            }
        }
        result = selector(mock_db, "test question")
        self.assertEqual(result.selected_tables, {"TableA"})
        self.assertEqual(result.db_name, "test_db")
        self.assertEqual(result.question, "test question")
        self.assertEqual(result.available_tables, {"TableA", "TableB"})

    def test_call_with_empty_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            generations=[
                [MagicMock(text=" ")]  # Set text to an empty string
            ]
        )
        selector = CoreTableSelector(
            llm=mock_llm, prompt=_TableSelectorPrompts().LANGCHAIN_DECIDER_PROMPT
        )
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.db._usable_tables = {"TableA", "TableC"}
        mock_db.descriptor = {
            "TableA": "Description A",
            "TableB": "Description B",
            "TableC": "Description C",
        }
        # with self.assertLogs("nl2sql.tasks.table_selection.core", level="CRITICAL"):
        result = selector(mock_db, "test question")
        self.assertEqual(result.selected_tables, set())
        self.assertEqual(result.db_name, "test_db")
        self.assertEqual(result.question, "test question")
        self.assertEqual(result.available_tables, {"TableA", "TableC"})