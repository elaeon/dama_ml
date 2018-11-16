import unittest
import numpy as np
import pandas as pd
import psycopg2

from ml.data.db import SQL
from ml import fmtypes


def get_column(data, column_index):
    column = []
    for row in data:
        column.append(row[column_index])
    return column


class TestSQL(unittest.TestCase):
    def setUp(self):
        self.data = np.asarray([
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
        ], dtype="|O")
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                sql.build_schema(columns=[("X0", fmtypes.TEXT), ("X1", fmtypes.ORDINAL),
                    ("X2", fmtypes.DENSE)], indexes=["X1"])
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
                self.assertEqual((sql[["x0", "x1"]].to_ndarray() == self.data[:, :2]).all(), True)
                self.assertEqual((sql[3:8].to_ndarray() == self.data[3:8]).all(), True)
                self.assertEqual((sql[3:].to_ndarray() == self.data[3:]).all(), True)
                self.assertEqual((sql[1].to_ndarray() == self.data[1]).all(), True)
                self.assertEqual((sql[5].to_ndarray() == self.data[5]).all(), True)
                self.assertEqual((sql[:10].to_ndarray() == self.data[:10]).all(), True)
                self.assertEqual((sql[["x1", "x2"]][:5].to_ndarray() == self.data[:5, 1:]).all(), True)
        except psycopg2.OperationalError:
            pass

    def test_to_df(self):
        try:
            df = pd.DataFrame(self.data, columns=["x0", "x1", "x2"])
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertEqual((sql["x0"].to_df().values.reshape(-1) == df["x0"].values).all(), True)
        except psycopg2.OperationalError:
            pass

    def test_batchs(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                self.assertEqual((sql[2:6].to_ndarray() == self.data[2:6]).all(), True)
        except psycopg2.OperationalError:
            pass

    def test_dtypes(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                dtypes = dict(sql.dtypes)
                self.assertEqual(list(dtypes.keys()), ['x0', 'x1', 'x2'])
                self.assertEqual(list(dtypes.values()), ['|O', 'int', 'float'])
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
                self.assertEqual((sql[3].to_ndarray()[0][0] == "0"), True)
                self.assertEqual((sql[3].to_ndarray()[0][1] == 0), True)
                self.assertEqual((sql[3].to_ndarray()[0][2] == 0), True)
        except psycopg2.OperationalError:
            pass

    def test_insert_update_by_index(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test") as sql:
                sql[12] = ["0", 0, 0]
                self.assertEqual((sql[12].to_ndarray()[0][0] == "0"), True)
                self.assertEqual((sql[12].to_ndarray()[0][1] == 0), True)
                self.assertEqual((sql[12].to_ndarray()[0][2] == 0), True)
                values = [["k", 11, 1.1], ["l", 12, 1.2], ["m", 13, 1.3]]
                sql[10:] = values
                sql_values = sql[10:].to_ndarray()
                self.assertCountEqual(sql_values[0], values[0])
                self.assertCountEqual(sql_values[1], values[1])
                self.assertCountEqual(sql_values[2], values[2])

                values = [["A", 1, 1], ["B", 2, 2], ["C", 3, 3]]
                sql[:3] = values
                sql_values = sql[:3].to_ndarray()
                self.assertCountEqual(sql_values[0], values[0])
                self.assertCountEqual(sql_values[1], values[1])
                self.assertCountEqual(sql_values[2], values[2])

                values = [["M", 13, 13], ["N", 14, 1.4], ["O", 15, 1.5]]
                sql[12:14] = values
                sql_values = sql[12:15].to_ndarray()
                self.assertCountEqual(sql_values[0], values[0])
                self.assertCountEqual(sql_values[1], values[1])
                self.assertCountEqual(sql_values[2], values[2])
        except psycopg2.OperationalError:
            pass

    def test_index_limit(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"]) as sql:
                self.assertCountEqual(sql[:3].reshape(-1), ["a", "b", "c"])
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["B", "C"]) as sql:
                self.assertCountEqual(sql[:3].reshape(-1), [1, 0.1, 2, 0.2, 3, 0.3])
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A", "B", "C"]) as sql:
                self.assertCountEqual(sql[:3].reshape(-1), ["a", 1, 0.1, "b", 2, 0.2, "c", 3, 0.3])
        except psycopg2.OperationalError:
            pass

    def test_random(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by="rand") as sql:
                sql[:10]
                self.assertCountEqual(sql.query, "SELECT a FROM test ORDER BY random() LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_sample(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by="rand") as sql:
                self.assertEqual(len(sql.reader().sample(5).to_memory()), 5)
        except psycopg2.OperationalError:
            pass

    def test_order_by(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"]) as sql:
                sql[:10]
                self.assertCountEqual(sql.query, "SELECT a FROM test ORDER BY id LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_no_order(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by=None) as sql:
                sql[:10]
                self.assertCountEqual(sql.query, "SELECT a FROM test  LIMIT 10")
        except psycopg2.OperationalError:
            pass

    def test_no_order_no_limit(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", only=["A"], order_by=None) as sql:
                sql[:]
                self.assertCountEqual(sql.query, "SELECT a FROM test  ")
        except psycopg2.OperationalError:
            pass

    def test_slide_col(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test", order_by=None) as sql:
                sql[:, 1]
                self.assertCountEqual(sql.query, "SELECT b FROM test  ")
        except psycopg2.OperationalError:
            pass


class TestSQLDateTime(unittest.TestCase):
    def setUp(self):
        self.data = [
            ["a", "2018-01-01 08:31:28"],
            ["b", "2018-01-01 09:31:28"],
            ["c", "2018-01-01 10:31:28"],
            ["d", "2018-01-01 11:31:28"],
            ["e", "2018-01-01 12:31:28"],
            ["f", "2018-01-01 13:31:28"],
            ["g", "2018-01-01 14:31:28"],
            ["h", "2018-01-01 15:31:28"]
        ]
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test_dt") as sql:
                sql.build_schema(columns=[("A", fmtypes.TEXT), ("B", fmtypes.DATETIME)])
                sql.insert(self.data)
        except psycopg2.OperationalError:
            pass

    def tearDown(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test_dt") as sql:
                sql.destroy()
        except psycopg2.OperationalError:
            pass

    def test_data_df(self):
        try:
            with SQL(username="alejandro", db_name="ml", table_name="test_dt",
                df=True) as sql:
                df = sql["B"]
                self.assertEqual(str(df.dtypes[0]), "datetime64[ns]")
        except psycopg2.OperationalError:
            pass


if __name__ == '__main__':
    unittest.main()
