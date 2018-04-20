import unittest
import numpy as np
import pandas as pd
import psycopg2

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
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                sql.build_schema(columns=[("A", fmtypes.TEXT), ("B", fmtypes.ORDINAL), 
                    ("C", fmtypes.DENSE)], indexes=["B"])
                sql.insert(self.data)
        except psycopg2.OperationalError:
            pass

    def tearDown(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                sql.destroy()
        except psycopg2.OperationalError:
            pass

    def test_index(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertItemsEqual(sql[1].to_memory()[0], self.data[1])
                self.assertItemsEqual(sql[5].to_memory()[0], self.data[5])
                self.assertItemsEqual(sql[1:].to_memory()[2], self.data[1:][2])
                self.assertItemsEqual(sql[:10].to_memory()[5], self.data[:10][5])
                self.assertItemsEqual(sql[3:8].to_memory()[1], self.data[3:8][1])
        except psycopg2.OperationalError:
            pass
 
    def test_key(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", chunks_size=2) as sql:
                self.assertItemsEqual(sql["A"].flat().to_memory(), get_column(self.data, 0))
                self.assertItemsEqual(sql["B"].flat().to_memory(), get_column(self.data, 1))
                self.assertItemsEqual(sql["C"].flat().to_memory(), get_column(self.data, 2))
        except psycopg2.OperationalError:
            pass

    def test_multikey(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertItemsEqual(sql[["A", "B"]].to_memory()[0], ['a', 1])
                self.assertItemsEqual(sql[["B", "C"]].to_memory()[0], [1, 0.1])
                self.assertItemsEqual(sql[["A", "C"]].to_memory()[0], ['a', 0.1])
        except psycopg2.OperationalError:
            pass

    def test_data_df(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test",
                chunks_size=12, df=True) as sql:
                self.assertEqual(type(sql["A"].to_memory()), pd.DataFrame)
        except psycopg2.OperationalError:
            pass

    def test_chunks(self):
        try:
            with SQL(username="alejandro", db_name="ml", order_by=['id'],
                table_name="test", chunks_size=3, df=False) as sql:
                self.assertItemsEqual(sql[2:6].flat().to_memory()[0], np.asarray(self.data[2:6])[0][0])
        except psycopg2.OperationalError:
            pass

    def test_columns(self):
        try:
            with SQL(username="alejandro", db_name="ml",
                table_name="test", chunks_size=12, df=True) as sql:
                columns = sql.columns()
                self.assertEqual(columns.keys(), ['id', 'a', 'b', 'c'])
                self.assertEqual(columns.values(), ['int', '|O', 'int', 'float'])
        except psycopg2.OperationalError:
            pass

    def test_shape(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertEqual(sql.shape, (12, 3))

            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertEqual(sql[3:].shape, (9, 3))
        except psycopg2.OperationalError:
            pass

    def test_update(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                sql.update(3, ("0", 0, 0))
                self.assertItemsEqual(sql[3].flat().to_memory(), ["0", 0, 0])
        except psycopg2.OperationalError:
            pass

    def test_insert_update_by_index(self):
        try:
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
        except psycopg2.OperationalError:
            pass

    def test_index_limit(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"]) as sql:
                self.assertItemsEqual(sql[:3].flat().to_memory(), ["a", "b", "c"])
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["B", "C"]) as sql:
                self.assertItemsEqual(sql[:3].flat().to_memory(), [1, 0.1, 2, 0.2, 3, 0.3])
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A", "B", "C"]) as sql:
                self.assertItemsEqual(sql[:3].flat().to_memory(), ["a", 1, 0.1, "b", 2, 0.2, "c", 3, 0.3])
        except psycopg2.OperationalError:
            pass

    def test_random(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by="rand") as sql:
                sql[:10].to_memory()
                self.assertItemsEqual(sql.query, "SELECT a FROM test ORDER BY random() LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_sample(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by="rand") as sql:
                self.assertEqual(len(sql[:].sample(5).to_memory()), 5)
        except psycopg2.OperationalError:
            pass

    def test_order_by(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"]) as sql:
                sql[:10].to_memory()
                self.assertItemsEqual(sql.query, "SELECT a FROM test ORDER BY id LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_no_order(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by=None) as sql:
                sql[:10].to_memory()
                self.assertItemsEqual(sql.query, "SELECT a FROM test  LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_no_order_no_limit(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by=None) as sql:
                sql[:].to_memory()
                self.assertItemsEqual(sql.query, "SELECT a FROM test  ")
        except psycopg2.OperationalError:
            pass


if __name__ == '__main__':
    unittest.main()
