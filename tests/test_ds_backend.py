import unittest
import numpy as np

from ml.extractors.db import SQL
from ml import fmtypes


class TestSQL(unittest.TestCase):
    def setUp(self):
        self.X = np.random.rand(100, 10)
        self.Y = np.random.rand(100)

    def tearDown(self):
        pass

    def test_build_schema(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            print(sql.exists())
            #sql.build_schema(columns_name=["A", "B", "C"], fmtypes=[fmtypes.TEXT, fmtypes.ORDINAL, fmtypes.DENSE], indexes="B")

    def test_data(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="holdings_air_hpg_cal", limit=100, chunks_size=12) as sql:
            print(sql["visitors"].flat().to_narray())

    def test_data_df(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="holdings_air_hpg_cal", limit=100, chunks_size=12, columns_name=True) as sql:
            print(sql["visitors"].to_df())

    def test_slice(self):
        with SQL(username="alejandro", db_name="ml", order_by=['store_id'],
            table_name="holdings_air_hpg_cal", limit=100, chunks_size=12, columns_name=False) as sql:
            print(sql[2:6].to_memory())

    def test_columns(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="holdings_air_hpg_cal", limit=100, chunks_size=12, columns_name=True) as sql:
            print(sql.columns())

    def test_shape(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="holdings_air_hpg_cal", limit=100) as sql:
            self.assertEqual(sql.shape, (100, 10))

        with SQL(username="alejandro", db_name="ml",
            table_name="holdings_air_hpg_cal", limit=None) as sql:
            #self.assertEqual(sql.shape, (100, 10))
            print(sql.shape)


if __name__ == '__main__':
    unittest.main()
