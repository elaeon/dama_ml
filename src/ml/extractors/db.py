import csv
from ml import fmtypes
#from tqdm import tqdm
import psycopg2
from ml.layers import IterLayer


class DBConection(object):
    def __init__(self, username, db_name):
        self.username = username
        self.db_name = db_name

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.db_name, 
                                                        username=self.username))
        return self.conn

    def __exit__(self, type, value, traceback):
        self.conn.close()


class SQL(object):
    def __init__(self, username, db_name, table_name, columns=None, flat=False):
        self.db_conn = DBConection(username, db_name)
        self.table_name = table_name
        self.columns = columns
        self.limit = None
        self.flat = flat

    @property
    def shape(self):
        if self.limit is not None:
            if self.flat is True:
                return (self.limit,)
            else:
                return self.limit, len(self.columns)

        with self.db_conn as conn:
            cur = conn.cursor()
            query = "SELECT COUNT(*) FROM {table_name}".format(
                table_name=self.table_name)
            cur.execute(query)
            size = cur.fetchone()[0]
            cur.close()

        if self.flat is True:
            return (size,)
        else:
            return size, len(self.columns)

    def format_columns(self, columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)

    def stream(self, limit=None):
        self.limit = limit
        if self.limit is None:
            limit = ""
        else:
            limit = "LIMIT {}".format(self.limit)

        conn = self.db_conn.__enter__()
        #with self.db_conn as conn:
        cur = conn.cursor()
        query = "SELECT {columns} FROM {table_name} {limit}".format(
            columns=self.format_columns(self.columns), table_name=self.table_name, 
            limit=limit)
        cur.execute(query)
        return IterLayer(cur, shape=self.shape)
