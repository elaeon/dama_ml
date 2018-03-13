import csv
from ml import fmtypes
import psycopg2
from ml.layers import IterLayer
import numpy as np


class SQL(object):
    def __init__(self, username, db_name, table_name, order_by=None, limit=None, 
        chunks_size=0, columns_name=False):
        self.conn = None
        self.username = username
        self.db_name = db_name
        self.table_name = table_name
        self.limit = limit
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
        cur = self.conn.cursor("sqlds", scrollable=False, withhold=False)
        columns = self.columns()

        if not isinstance(key, list) and not isinstance(key, slice):
            _columns = [key]
            num_features = len(_columns)
            size = self.shape[0]
            start = 0
        elif isinstance(key, slice):
            num_features = len(columns)
            _columns = columns.keys()
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.limit
            else:
                stop = key.stop
                self.limit = key.stop
                self._build_limit_text()

            size = abs(start - stop)

        if self.columns_name is True:
            self.dtype = [(column_name, columns[column_name]) for column_name in _columns]
        else:
            self.dtype = None

        query = "SELECT {columns} FROM {table_name} {order_by} {limit}".format(
            columns=self.format_columns(_columns), table_name=self.table_name,
            order_by=self.order_by_txt,
            limit=self.limit_txt)
        cur.execute(query)
        #cur.itersize = 2000
        cur.scroll(start)
        it = IterLayer(cur, shape=(size, num_features), dtype=self.dtype)
        if self.chunks_size is not None:
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
        if self.limit is not None:
            return self.limit, self.num_columns()
        else:
            cur = self.conn.cursor()
            query = "SELECT COUNT(*) FROM {table_name}".format(
                table_name=self.table_name)
            cur.execute(query)
            size = cur.fetchone()[0]
            return size, self.num_columns()

    def num_columns(self):
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM information_schema.columns WHERE table_name=%(table_name)s"
        cur.execute(query, {"table_name": self.table_name})
        return cur.fetchone()[0]

    def columns(self):
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s"
        cur.execute(query, {"table_name": self.table_name})
        columns = {}
        types = {"text": "|O", "integer": "int"}
        for column in cur.fetchall():
            columns[column[3]] = types.get(column[7], "|O")
        return columns

    def format_columns(self, columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)

    def build_schema(self, columns_name, fmtypes, indexes):
        
        def build():
            columns_types = ["id serial PRIMARY KEY"]
            for col, fmtype in zip(columns_name, fmtypes):
                columns_types.append("{col} {type}".format(col=col, type=fmtype.db_type))
            cols = "("+", ".join(columns_types)+")"
        #    index = "CREATE INDEX {id_name}_{name}_index ON {name} ({id_name})".format(name=table_name, id_name=id_name)
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE {name}
                {columns};
            """.format(name=self.table_name, columns=cols))
            #cur.execute(index)

        if not self.exists():
            build()        
        #else:
        #    cur.execute("DROP TABLE {name};".format(name=table_name))
        #    build()
        self.conn.commit()
        #cur.close()
        #conn.close()

    def exists(self):
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=self.table_name))
        return cur.fetchone()[0]

    def destroy(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name};".format(name=self.table_name))
        self.conn.commit()
