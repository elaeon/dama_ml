import csv
from ml import fmtypes
import psycopg2
from ml.layers import IterLayer
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import uuid


class SQL(object):
    def __init__(self, username, db_name, table_name, order_by=None, 
        chunks_size=0, columns_name=False):
        self.conn = None
        self.cur = None
        self.username = username
        self.db_name = db_name
        self.table_name = table_name
        self.limit = None
        self.order_by = order_by
        self._build_order_text()
        self._build_limit_text()
        self.chunks_size = chunks_size
        self.columns_name = columns_name

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.db_name, 
                                                        username=self.username))
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        self.cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        columns = self.columns(exclude_id=True)

        if isinstance(key, str):
            _columns = [key]
            num_features = len(_columns)
            size = self.shape[0]
            start = 0
        elif isinstance(key, list):
            _columns = key
            num_features = len(_columns)
            size = self.shape[0]
            start = 0
        elif isinstance(key, int):
            _columns = columns.keys()
            num_features = len(_columns)
            size = 1
            start = key
        elif isinstance(key, slice):
            _columns = columns.keys()
            num_features = len(_columns)
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.shape[0]
            else:
                stop = key.stop
                self.limit = key.stop
                self._build_limit_text()

            size = abs(start - stop)

        if self.columns_name is True:
            self.dtype = [(column_name, columns[column_name.lower()]) for column_name in _columns]
        else:
            self.dtype = None

        query = "SELECT {columns} FROM {table_name} {order_by} {limit}".format(
            columns=self.format_columns(_columns), table_name=self.table_name,
            order_by=self.order_by_txt,
            limit=self.limit_txt)
        self.cur.execute(query)
        #cur.itersize = 2000
        self.cur.scroll(start)
        it = IterLayer(self.cur, shape=(size, num_features), dtype=self.dtype)
        if self.chunks_size is not None and self.chunks_size > 0:
            return it.to_chunks(self.chunks_size)
        return it

    def __setitem__(self, key, value):
        pass

    def _build_limit_text(self):
        if self.limit is None:
            self.limit_txt = ""
        else:
            self.limit_txt = "LIMIT {}".format(self.limit)

    def _build_order_text(self):
        if self.order_by is None:
            self.order_by_txt = ""
        else:
            self.order_by_txt = "ORDER BY " + ",".join(self.order_by)

    def close(self):
        self.conn.close()

    @property
    def shape(self):
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM {table_name}".format(
            table_name=self.table_name)
        cur.execute(query)
        size = cur.fetchone()[0]
        return size, self.num_columns(exclude_id=True)

    def num_columns(self, exclude_id=False):
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM information_schema.columns WHERE table_name=%(table_name)s"
        cur.execute(query, {"table_name": self.table_name})
        if exclude_id is True:
            return cur.fetchone()[0] - 1
        else:
            return cur.fetchone()[0]

    def columns(self, exclude_id=False):
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s ORDER BY ordinal_position"
        cur.execute(query, {"table_name": self.table_name})
        columns = OrderedDict()
        types = {"text": "|O", "integer": "int", "double precision": "float"}
        for column in cur.fetchall():
            columns[column[3]] = types.get(column[7], "|O")
        if exclude_id is True:
            del columns["id"]
        return columns

    def format_columns(self, columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)

    def build_schema(self, columns, indexes=None):
        def build():
            columns_types = ["id serial PRIMARY KEY"]
            for col, fmtype in columns:
                columns_types.append("{col} {type}".format(col=col, type=fmtype.db_type))
            cols = "("+", ".join(columns_types)+")"
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE {name}
                {columns};
            """.format(name=self.table_name, columns=cols))
            if isinstance(indexes, list):
                for index in indexes:
                    if isinstance(index, tuple):
                        index_columns = ",".join(index)
                        index_name = "_".join(index)
                    else:
                        index_columns = index
                        index_name = index
                    index_q = "CREATE INDEX {i_name}_{name}_index ON {name} ({i_columns})".format(
                        name=self.table_name, i_name=index_name, i_columns=index_columns)
                    cur.execute(index_q)

        if not self.exists():
            build()
        self.conn.commit()

    def insert(self, data):
        header = self.columns(exclude_id=True)
        columns = "("+", ".join(header)+")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(name=self.table_name, columns=columns)
        values_str = "("+", ".join(["%s" for _ in range(len(header))])+")"
        insert = insert_str+" "+values_str
        cur = self.conn.cursor()
        for row in tqdm(data):
            cur.execute(insert, row)
        self.conn.commit()

    def exists(self):
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=self.table_name))
        return cur.fetchone()[0]

    def destroy(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name};".format(name=self.table_name))
        self.conn.commit()
