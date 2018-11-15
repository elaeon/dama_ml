import psycopg2
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from ml.data.it import BaseIterator
from ml.data.abc import AbsDataset
from ml.utils.decorators import cache
from ml.utils.numeric_functions import max_dtype
from ml.utils.basic import StructArray


class IteratorConn(BaseIterator):
    def __init__(self, conn, length=None, dtypes=None, shape=None, type_elem=None,
                 batch_size: int=0, table_name=None) -> None:
        self.conn = conn
        self.batch_size = batch_size
        self.table_name = table_name
        super(IteratorConn, self).__init__(None, length=length, dtypes=dtypes, shape=shape, type_elem=type_elem)

    def __getitem__(self, key):
        cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        stop = None
        if isinstance(key, tuple):
            _columns = self.labels()[key[1]]
            key = key[0]
        else:
            _columns = None

        if isinstance(key, str):
            if _columns is None:
                _columns = [key]
            start = 0
        elif isinstance(key, list):
            if _columns is None:
                _columns = key
            start = 0
        elif isinstance(key, int):
            if _columns is None:
                _columns = self.labels
            start = key
            stop = start + 1
        elif isinstance(key, slice):
            if _columns is None:
                _columns = self.labels
            if key.start is None:
                start = 0
            else:
                start = key.start

            if key.stop is None:
                stop = self.shape[0]
            else:
                stop = key.stop
        elif key is None:
            start = 0

        if stop is None:
            limit_txt = ""
        else:
            limit_txt = "LIMIT {}".format(stop)

        query = "SELECT {columns} FROM {table_name} {order_by} {limit}".format(
            columns=self.format_columns(_columns), table_name=self.table_name,
            order_by="id",
            limit=limit_txt)
        cur.execute(query)
        cur.itersize = self.batch_size
        cur.scroll(start)
        e = np.asarray(cur.fetchall())
        #print("****", e.shape, e.dtype, self.dtype, self.dtypes, self.shape)
        return e

    def format_columns(self, columns):
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
        self._build_order_text()
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

    def __exit__(self, type, value, traceback):
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
                dtypes = [(col, dtypes[1]) for col, dtypes in zip(value, self.dtypes)]
            else:
                raise Exception
        self.dtypes = dtypes

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            #_columns = [key]
            #num_features = len(_columns)
            #size = self.shape[0]
            #start = 0
            pass
        elif isinstance(key, list):
            pass
            #_columns = key
            #num_features = len(_columns)
            #size = self.shape[0]
            #start = 0
        elif isinstance(key, int):
            if key >= self.last_id():
                self.insert([value])
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
            elif start <= last_id and stop >= last_id:
                size = abs(start - last_id)
                i = start
                for row in value[:size]:
                    self.update(i, row)
                    i += 1
                self.insert(value[size:])

    def _build_order_text(self):
        if self.order_by is None:
            self.order_by_txt = ""
        else:
            if "rand" in self.order_by:
                self.order_by_txt = "ORDER BY random()"
            else:
                self.order_by_txt = "ORDER BY " + ",".join(self.order_by)

    def close(self):
        self.conn.close()

    @property
    @cache
    def shape(self):
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
                 "double precision": np.dtype("float"), "boolean": np.dtype("bool")}
        if self.only_columns is None:                
            for column in cur.fetchall():
                dtypes[column[3]] = types.get(column[7], np.dtype("object"))
            #if exclude_id is True:
            if "id" in dtypes:
                del dtypes["id"]
        else:
            for column in cur.fetchall():
                if column[3] in self.only_columns:
                    dtypes[column[3]] = types.get(column[7], np.dtype("object"))

        if len(dtypes) == 0:
            return None
        else:
            return list(dtypes.items())

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

    def insert(self, data, chunks_size=258):
        from psycopg2.extras import execute_values
        from ml.utils.seq import grouper_chunk
        header = self.labels
        columns = "("+", ".join(header)+")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(
            name=self.table_name, columns=columns)
        #values_str = "("+", ".join(["%s" for _ in range(len(header))])+")"
        #insert = insert_str+" "+values_str
        insert = insert_str + " %s"
        cur = self.conn.cursor()
        for row in tqdm(grouper_chunk(chunks_size, data)):
            #cur.execute(insert, cache_row)
            execute_values(cur, insert, row, page_size=chunks_size)
        self.conn.commit()

    def update(self, id, values):
        header = self.columns
        columns = "("+", ".join(header)+")"
        update_str = "UPDATE {name} SET {columns} = {values} WHERE id = {id}".format(name=self.table_name, 
            columns=columns, values=tuple(values), id=id+1)
        cur = self.conn.cursor()
        cur.execute(update_str)
        self.conn.commit()

    def exists(self):
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=self.table_name))
        return cur.fetchone()[0]

    def destroy(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name};".format(name=self.table_name))
        self.conn.commit()

    def to_ndarray(self) -> np.ndarray:
        return self.data.to_ndarray()

    def to_df(self) -> pd.DataFrame:
        return self.data.to_df()