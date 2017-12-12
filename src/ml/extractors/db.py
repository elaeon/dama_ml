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
    def __init__(self, username, db_name, table_name, columns=None):
        self.db_conn = DBConection(username, db_name)
        self.table_name = table_name
        self.columns = self.format_columns(columns)

    @property
    def shape(self):
        with self.db_conn as conn:
            cur = conn.cursor()
            query = "SELECT COUNT(*) FROM {table_name}".format(
                table_name=self.table_name)
            cur.execute(query)
            shape  = cur.fetchone()
            cur.close()
        return shape

    def format_columns(self, columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)

    def stream(self, fmtypes=None):
        #if fmtypes is None:
        #    fmtypes = [fmtypes.TEXT]*len(header)

        shape = self.shape
        with self.db_conn as conn:
            cur = conn.cursor()
            query = "SELECT {columns} FROM {table_name}".format(
                columns=self.columns, table_name=self.table_name)
            cur.execute(query)
            return IterLayer(cur, shape=shape)
