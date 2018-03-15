import unittest
import numpy as np

from ml.extractors.db import SQL
from ml import fmtypes


class TestSQL(unittest.TestCase):
    def setUp(self):
        self.data = [
            ["a", 1, 0.1],
            ["b", 2, 0.2],
            ["c", 3, 0.3],
            ["d", 4, 0.4],
            ["e", 5, 0.5],
            ["f", 6, 0.6],
            ["g", 7, 0.7],
            ["h", 8, 0.8],
            ["i", 9, 0.9],
            ["j", 10, 1],
            ["k", 11, 1.1],
            ["l", 12, 1.2],
        ]
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            sql.build_schema(columns=[("A", fmtypes.TEXT), ("B", fmtypes.ORDINAL), 
                ("C", fmtypes.DENSE)], indexes=["B"])
            sql.insert(self.data)

    def tearDown(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            sql.destroy()

    def test_index(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="test") as sql:
            self.assertItemsEqual(sql[1].to_narray()[0], self.data[1])
            self.assertItemsEqual(sql[5].to_narray()[0], self.data[5])
            self.assertItemsEqual(sql[1:].to_narray()[2], self.data[1:][2])
            self.assertItemsEqual(sql[:10].to_narray()[5], self.data[:10][5])
            self.assertItemsEqual(sql[3:8].to_narray()[1], self.data[3:8][1])
            #print(sql["visitors"].flat().to_narray())

    def test_data_df(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="test", limit=100, chunks_size=12, columns_name=True) as sql:
            print(sql["visitors"].to_df())

    def test_slice(self):
        with SQL(username="alejandro", db_name="ml", order_by=['store_id'],
            table_name="test", limit=100, chunks_size=12, columns_name=False) as sql:
            print(sql[2:6].to_memory())

    def test_columns(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="test", limit=100, chunks_size=12, columns_name=True) as sql:
            print(sql.columns())

    def test_shape(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="test", limit=100) as sql:
            self.assertEqual(sql.shape, (100, 10))

        with SQL(username="alejandro", db_name="ml",
            table_name="test", limit=None) as sql:
            #self.assertEqual(sql.shape, (100, 10))
            print(sql.shape)


if __name__ == '__main__':
    unittest.main()
