import unittest
import numpy as np
import pandas as pd

from ml.db import SQL
from ml import fmtypes


def get_column(data, column_index):
    column = []
    for row in data:
        column.append(row[column_index])
    return column


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
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            self.assertItemsEqual(sql[1].to_narray()[0], self.data[1])
            self.assertItemsEqual(sql[5].to_narray()[0], self.data[5])
            self.assertItemsEqual(sql[1:].to_narray()[2], self.data[1:][2])
            self.assertItemsEqual(sql[:10].to_narray()[5], self.data[:10][5])
            self.assertItemsEqual(sql[3:8].to_narray()[1], self.data[3:8][1])
 
    def test_key(self):
        with SQL(username="alejandro", db_name="ml", table_name="test", chunks_size=2) as sql:
            self.assertItemsEqual(sql["A"].flat().to_narray(), get_column(self.data, 0))
            self.assertItemsEqual(sql["B"].flat().to_narray(), get_column(self.data, 1))
            self.assertItemsEqual(sql["C"].flat().to_narray(), get_column(self.data, 2))

    def test_multikey(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            self.assertItemsEqual(sql[["A", "B"]].to_narray()[0], ['a', 1])
            self.assertItemsEqual(sql[["B", "C"]].to_narray()[0], [1, 0.1])
            self.assertItemsEqual(sql[["A", "C"]].to_narray()[0], ['a', 0.1])

    def test_data_df(self):
        with SQL(username="alejandro", db_name="ml", table_name="test",
            chunks_size=12, columns_name=True) as sql:
            self.assertEqual(type(sql["A"].to_df()), pd.DataFrame)

    def test_chunks(self):
        with SQL(username="alejandro", db_name="ml", order_by=['id'],
            table_name="test", chunks_size=3, columns_name=False) as sql:
            self.assertItemsEqual(sql[2:6].flat().to_memory()[0], np.asarray(self.data[2:6])[0][0])

    def test_columns(self):
        with SQL(username="alejandro", db_name="ml",
            table_name="test", chunks_size=12, columns_name=True) as sql:
            columns = sql.columns()
            self.assertEqual(columns.keys(), ['id', 'a', 'b', 'c'])
            self.assertEqual(columns.values(), ['int', '|O', 'int', 'float'])

    def test_shape(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            self.assertEqual(sql.shape, (12, 3))

        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            self.assertEqual(sql[3:].shape, (9, 3))

    def test_update(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            sql.update(3, ("0", 0, 0))
            self.assertItemsEqual(sql[3].flat().to_memory(), ["0", 0, 0])

    def test_insert_update_by_index(self):
        with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
            sql[12] = ["0", 0, 0]
            self.assertItemsEqual(sql[12].to_memory()[0], ["0", 0, 0])
            sql[10] = ["0", 0, 0]
            self.assertItemsEqual(sql[10].to_memory()[0], ["0", 0, 0])
            values = [["k", 11, 1.1], ["l", 12, 1.2], ["m", 13, 1.3]]
            sql[10:] = values
            sql_values = sql[10:].to_memory()
            self.assertItemsEqual(sql_values[0], values[0])
            self.assertItemsEqual(sql_values[1], values[1])
            self.assertItemsEqual(sql_values[2], values[2])

            values = [["A", 1, 1], ["B", 2, 2], ["C", 3, 3]]
            sql[:3] = values
            sql_values = sql[:3].to_memory()
            self.assertItemsEqual(sql_values[0], values[0])
            self.assertItemsEqual(sql_values[1], values[1])
            self.assertItemsEqual(sql_values[2], values[2])

            values = [["M", 13, 13], ["N", 14, 1.4], ["O", 15, 1.5]]
            sql[12:14] = values
            sql_values = sql[12:15].to_memory()
            self.assertItemsEqual(sql_values[0], values[0])
            self.assertItemsEqual(sql_values[1], values[1])
            self.assertItemsEqual(sql_values[2], values[2])


if __name__ == '__main__':
    unittest.main()
