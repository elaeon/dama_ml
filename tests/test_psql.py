import unittest
import numpy as np
import psycopg2

from ml.utils.basic import Login
from ml.data.it import Iterator
from ml.data.db import Schema
from ml.data.ds import Data


def get_column(data, column_index):
    column = []
    for row in data:
        column.append(row[column_index])
    return column


class TestSQL(unittest.TestCase):
    def setUp(self):
        self.login = Login(username="alejandro", resource="ml")

    def tearDown(self):
        pass

    def test_schema(self):
        try:
            with Schema(login=self.login) as schema:
                schema.build("test_schema_db", [("c0", np.dtype("O"))])
                self.assertEqual(schema.exists("test_schema_db"), True)
                schema.destroy("test_schema_db")
                self.assertEqual(schema.exists("test_schema_db"), False)
        except psycopg2.OperationalError:
            pass

    def test_schema_info(self):
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype(bool)), ("x2", np.dtype(int)),
                  ("x3", np.dtype(float)), ("x4", np.dtype("datetime64[ns]"))]
        try:
            with Schema(login=self.login) as schema:
                schema.build("test_schema_db", dtypes)
                self.assertEqual(dtypes, schema.info("test_schema_db"))
                schema.destroy("test_schema_db")
        except psycopg2.OperationalError:
            pass

    def test_insert(self):
        data = np.asarray([
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
        ], dtype="O")
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype(int)), ("x2", np.dtype(float))]
        try:
            with Schema(login=self.login) as schema:
                schema.build("test_schema_db", dtypes)
                schema.insert("test_schema_db", Iterator(data).batchs(batch_size=10, batch_type="array"))
                schema.destroy("test_schema_db")
        except psycopg2.OperationalError:
            pass

    def test_table(self):
        data = np.random.rand(10, 2)
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype(object))]
        try:
            with Schema(login=self.login) as schema:
                schema.build("test_schema_db", dtypes)
                schema.insert("test_schema_db", Iterator(data).batchs(batch_size=10, batch_type="array"))
                self.assertEqual(schema["test_schema_db"].shape, (10, 2))
                self.assertEqual(schema["test_schema_db"].last_id(), 10)
                schema.destroy("test_schema_db")
        except psycopg2.OperationalError:
            pass

    def test_datetime(self):
        data = [
            ["a", "2018-01-01 08:31:28"],
            ["b", "2018-01-01 09:31:28"],
            ["c", "2018-01-01 10:31:28"],
            ["d", "2018-01-01 11:31:28"],
            ["e", "2018-01-01 12:31:28"],
            ["f", "2018-01-01 13:31:28"],
            ["g", "2018-01-01 14:31:28"],
            ["h", "2018-01-01 15:31:28"]
        ]
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype("datetime64[ns]"))]
        try:
            with Schema(login=self.login) as schema:
                schema.build("test_schema_db", dtypes)
                schema.insert("test_schema_db", Iterator(data).batchs(batch_size=10, batch_type="array"))
                self.assertEqual(schema["test_schema_db"].shape, (8, 2))
                schema.destroy("test_schema_db")
        except psycopg2.OperationalError:
            pass

    def test_get(self):
        data = np.asarray([
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
        ], dtype="O")
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype(int)), ("x2", np.dtype(float))]
        try:
            with Schema(login=self.login) as schema:
                schema.destroy("test_schema_db")
                schema.build("test_schema_db", dtypes)
                schema.insert("test_schema_db", Iterator(data).batchs(batch_size=10, batch_type="array"))
                self.assertEqual((schema["test_schema_db"]["x1"][:5].compute().reshape(-1) == data[:5, 1]).all(), True)
                self.assertEqual((schema["test_schema_db"]["x0", "x1"].compute() == data[:, :2]).all(), True)
                self.assertEqual(schema["test_schema_db"]["x0"][3].compute(), data[3, 0])
                self.assertEqual((schema["test_schema_db"][1].compute().reshape(-1) == data[1]).all(), True)
                self.assertEqual((schema["test_schema_db"][1, 2, 3].compute() == data[[1, 2, 3]]).all(), True)
                self.assertEqual((schema["test_schema_db"]["x0"][1, 2, 3].compute().reshape(-1) == data[:, 0][[1, 2, 3]]).all(), True)
                schema.destroy("test_schema_db")
        except psycopg2.OperationalError:
            pass

    def test_set(self):
        data = np.asarray([
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
        ], dtype="O")
        dtypes = [("x0", np.dtype(object)), ("x1", np.dtype(int)), ("x2", np.dtype(float))]
        #try:
        with Schema(login=self.login) as schema:
            schema.destroy("test_schema_db")
            schema.build("test_schema_db", dtypes)
            schema["test_schema_db"][0:5] = data[0:5]
            schema["test_schema_db"][6:10] = data[6:10]
            print(schema["test_schema_db"][:].compute())
            schema.destroy("test_schema_db")
        #except psycopg2.OperationalError:
        #    pass

    def test_driver(self):
        with Schema(login=self.login) as schema:
            schema.destroy("test")
        x = np.random.rand(10)*100
        y = np.random.rand(10)*100
        data = Data(name="test", driver=Schema(login=self.login))
        data.from_data({"x": x, "y": y})
        with data:
            self.assertEqual((data["x"].to_ndarray(dtype=np.dtype("int8")) == x.astype("int8")).all(), True)
            self.assertEqual((data["y"].to_ndarray(dtype=np.dtype("int8")) == y.astype("int8")).all(), True)
        data.destroy()


if __name__ == '__main__':
    unittest.main()
