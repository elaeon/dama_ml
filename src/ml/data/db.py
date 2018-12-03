import psycopg2
import uuid
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import OrderedDict
from psycopg2.extras import execute_values
from ml.utils.seq import grouper_chunk
from ml.data.it import BaseIterator
from ml.abc.data import AbsDataset
from ml.utils.decorators import cache
from ml.utils.numeric_functions import max_dtype
from ml.utils.basic import StructArray
from ml.fmtypes import fmtypes_map


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


class SQL(AbsDataset):
    def __init__(self, username, db_name, table_name, only=None):
        self.conn = None
        self.username = username
        self.db_name = db_name
        self.table_name = table_name
        self.order_by = ["id"]
        self.dtype = None

        if only is not None:
            self.only_columns = set([column.lower() for column in only])
        else:
            self.only_columns = only

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.db_name,
                                                      username=self.username))
        self.conn.autocommit = False
        self.dtype = max_dtype(self.dtypes)
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def __next__(self):
        return NotImplemented

    def __iter__(self):
        return NotImplemented

    def batchs_writer(self, keys, data, init=0):
        return NotImplemented

    def url(self):
        return NotImplemented

    @property
    def data(self):
        labels_data = []
        dtypes = dict(self.dtypes)
        length = self.shape[0]
        for label in self.labels:
            dtype = [(label, dtypes[label])]
            it = IteratorConn(self.conn, dtypes=dtype, length=length, shape=(None, 1), table_name=self.table_name)
            labels_data.append((label, it))
        return StructArray(labels_data)

    @property
    def labels(self) -> list:
        return [c for c, _ in self.dtypes]

    @labels.setter
    def labels(self, value) -> None:
        with self:
            if len(value) == len(self.dtypes):
                self.dtypes = [(col, dtypes[1]) for col, dtypes in zip(value, self.dtypes)]
            else:
                raise Exception

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            pass
        elif isinstance(key, list):
            pass
        elif isinstance(key, int):
            if key >= self.last_id():
                self.from_data([value])
            else:
                self.update(key, value)
        elif isinstance(key, slice):
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.shape[0]
            else:
                stop = key.stop

            last_id = self.last_id()
            if stop <= last_id:
                i = start
                for row in value:
                    self.update(i, row)
                    i += 1
            elif start <= last_id <= stop:
                size = abs(start - last_id)
                i = start
                for row in value[:size]:
                    self.update(i, row)
                    i += 1
                self.from_data(value[size:])

    def close(self):
        self.conn.close()

    @property
    @cache
    def shape(self) -> tuple:
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM {table_name}".format(
            table_name=self.table_name)
        cur.execute(query)
        length = cur.fetchone()[0]
        return length, len(self.labels)

    def last_id(self):
        cur = self.conn.cursor()
        query = "SELECT last_value FROM {table_name}_id_seq".format(table_name=self.table_name)
        cur.execute(query)
        return cur.fetchone()[0]

    @property
    @cache
    def dtypes(self) -> list:
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s ORDER BY ordinal_position"
        cur.execute(query, {"table_name": self.table_name})
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

    @dtypes.setter
    def dtypes(self, values):
        pass

    def build_schema(self, columns, indexes=None):
        def build():
            columns_types = ["id serial PRIMARY KEY"]
            for col, dtype in columns:
                fmtype = fmtypes_map[dtype]
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

    def from_data(self, data, batch_size: int=258):
        header = self.labels
        columns = "(" + ", ".join(header) + ")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(
            name=self.table_name, columns=columns)
        insert = insert_str + " %s"
        cur = self.conn.cursor()
        for row in tqdm(grouper_chunk(batch_size, data)):
            execute_values(cur, insert, row, page_size=batch_size)
        self.conn.commit()

    def update(self, index, values):
        header = self.labels
        columns = "("+", ".join(header)+")"
        update_str = "UPDATE {name} SET {columns} = {values} WHERE id = {id}".format(
            name=self.table_name,
            columns=columns, values=tuple(values), id=index+1)
        cur = self.conn.cursor()
        cur.execute(update_str)
        self.conn.commit()

    def exists(self) -> bool:
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=self.table_name))
        return True if cur.fetchone()[0] else False

    def destroy(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name};".format(name=self.table_name))
        self.conn.commit()

    def to_ndarray(self) -> np.ndarray:
        return self.data.to_ndarray()

    def to_df(self) -> pd.DataFrame:
        return self.data.to_df()
