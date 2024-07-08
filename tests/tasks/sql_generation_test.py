import unittest
from unittest.mock import MagicMock
from nl2sql.tasks.sql_generation.core import CoreSqlGenerator, CoreSqlGenratorResult

class TestCoreSqlGenerator(unittest.TestCase):
    def test_core_sql_generator_with_valid_response(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            generations=[
                [
                    MagicMock(text="SELECT AVG(price) FROM products WHERE category = 'Electronics';")
                ]
            ]
        ) 

        mock_db = MagicMock()
        mock_db.db.dialect = "sqlite"
        mock_db.db.table_info = {
            "products": {"product_id": "INT PRIMARY KEY", "name": "TEXT", "price": "REAL", "category": "TEXT"}
        }
        mock_db.db._usable_tables = ["products"]
        mock_db.name = "test_db"
        mock_db.descriptor = "A test database"

        # Initialize with the mock LLM
        generator = CoreSqlGenerator(llm=mock_llm)

        # Run the generator
        result = generator(mock_db, "What is the average price of products in the 'Electronics' category?")

        # Assertions
        self.assertEqual(result.generated_query, "SELECT AVG(price) FROM products WHERE category = 'Electronics';")
        self.assertEqual(result.db_name, "test_db")
        self.assertEqual(result.question, "What is the average price of products in the 'Electronics' category?")
        self.assertEqual(len(result.intermediate_steps), 1)

        # Verify LLM call
        mock_llm.generate.assert_called_once() 