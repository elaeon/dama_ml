import psycopg2
import uuid
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
from psycopg2.extras import execute_values
from ml.data.it import BaseIterator
from ml.fmtypes import fmtypes_map
from ml.utils.basic import Shape, Login


class IteratorConn(BaseIterator):
    def __init__(self, conn, length=None, dtypes=None, shape=None, type_elem=None,
                 batch_size: int=0, table_name=None) -> None:
        self.conn = conn
        self.batch_size = batch_size
        self.table_name = table_name
        super(IteratorConn, self).__init__(None, length=length, dtypes=dtypes, shape=shape, type_elem=type_elem)

    def __getitem__(self, key):
        cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        start = 0
        stop = None
        if isinstance(key, tuple):
            _columns = self.labels[key[1]]
            key = key[0]
        else:
            _columns = None

        if isinstance(key, str):
            if _columns is None:
                _columns = [key]
        elif isinstance(key, list):
            if _columns is None:
                _columns = key
        elif isinstance(key, int):
            if _columns is None:
                _columns = self.labels
            start = key
            stop = start + 1
        elif isinstance(key, slice):
            if _columns is None:
                _columns = self.labels
            if key.start is not None:
                start = key.start

            if key.stop is None:
                stop = self.shape[0]
            else:
                stop = key.stop

        if stop is None:
            limit_txt = ""
        else:
            limit_txt = "LIMIT {}".format(stop)

        length = abs(stop - start)
        shape = [length] + list(self.shape[1:])
        query = "SELECT {columns} FROM {table_name} ORDER BY {order_by} {limit}".format(
            columns=IteratorConn.format_columns(_columns), table_name=self.table_name,
            order_by="id",
            limit=limit_txt)
        cur.execute(query)
        cur.itersize = self.batch_size
        cur.scroll(start)
        smx = np.empty(shape, dtype=self.dtype)
        for i, row in enumerate(cur.fetchall()):
            smx[i] = row
        return smx

    @staticmethod
    def format_columns(columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)


class Schema(object):
    def __init__(self, login: Login):
        self.login = login

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.login.resource, username=self.login.username))
        self.conn.autocommit = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def exists(self, table_name) -> bool:
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=table_name))
        return True if cur.fetchone()[0] else False

    def build(self, table_name, columns, indexes=None):
        def _build():
            columns_types = ["id serial PRIMARY KEY"]
            for col, dtype in columns:
                fmtype = fmtypes_map[dtype]
                columns_types.append("{col} {type}".format(col=col, type=fmtype.db_type))
            cols = "("+", ".join(columns_types)+")"
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE {name}
                {columns};
            """.format(name=table_name, columns=cols))
            if isinstance(indexes, list):
                for index in indexes:
                    if isinstance(index, tuple):
                        index_columns = ",".join(index)
                        index_name = "_".join(index)
                    else:
                        index_columns = index
                        index_name = index
                    index_q = "CREATE INDEX {i_name}_{name}_index ON {name} ({i_columns})".format(
                        name=table_name, i_name=index_name, i_columns=index_columns)
                    cur.execute(index_q)

        if not self.exists(table_name):
            _build()
        self.conn.commit()

    def destroy(self, table_name):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name}".format(name=table_name))
        self.conn.commit()

    def info(self, table_name):
        return self[table_name].info()

    def insert(self, table_name: str, data):
        header = self.info(table_name)
        columns = "(" + ", ".join([name for name, _ in header]) + ")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(
            name=table_name, columns=columns)
        insert = insert_str + " %s"
        cur = self.conn.cursor()
        for row in tqdm(data, total=data.num_splits()):
            execute_values(cur, insert, row, page_size=data.batch_size)
        self.conn.commit()

    def __getitem__(self, item):
        if isinstance(item, str):
            return Table(item, self.conn)


class Table(object):
    def __init__(self, name, conn):
        self.conn = conn
        self.name = name

    def query(self):
        return "SELECT * FROM {table_name}".format(table_name=self.name)

    @property
    def shape(self) -> Shape:
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM {table_name}".format(table_name=self.name)
        cur.execute(query)
        length = cur.fetchone()[0]
        return Shape({self.name: (length, len(self.info()))})

    def info(self):
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s ORDER BY ordinal_position"
        cur.execute(query, {"table_name": self.name})
        dtypes = OrderedDict()
        types = {"text": np.dtype("object"), "integer": np.dtype("int"),
                 "double precision": np.dtype("float"), "boolean": np.dtype("bool"),
                 "timestamp without time zone": np.dtype('datetime64[ns]')}

        for column in cur.fetchall():
            dtypes[column[3]] = types.get(column[7], np.dtype("object"))

        if "id" in dtypes:
            del dtypes["id"]

        if len(dtypes) > 0:
            return list(dtypes.items())

    def last_id(self):
        cur = self.conn.cursor()
        query = "SELECT last_value FROM {table_name}_id_seq".format(table_name=self.name)
        cur.execute(query)
        return cur.fetchone()[0]
