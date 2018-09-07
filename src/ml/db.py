import psycopg2
import uuid
from tqdm import tqdm
from collections import OrderedDict
from ml.data.it import Iterator
from ml.data.abc import AbsDataset
from ml.utils.decorators import cache


class SQL(AbsDataset):
    def __init__(self, username, db_name, table_name, order_by=["id"], 
        itersize=2000, df=False, only=None):
        self.conn = None
        self.cur = None
        self.username = username
        self.db_name = db_name
        self.table_name = table_name
        self.limit = None
        self.order_by = order_by
        self._build_order_text()
        self._build_limit_text()
        self.itersize = itersize
        self.df_dtype = df
        self.query = None

        if only is not None:
            self.only_columns = set([column.lower() for column in only])
        else:
            self.only_columns = only

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.db_name, 
                                                        username=self.username))
        self.conn.autocommit = False
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __next__(self):
        return NotImplemented

    def __iter__(self):
        return NotImplemented

    def chunks_writer(self, name, data, init=0):
        return NotImplemented

    def chunks_writer_split(self, data_key, labels_key, data, labels_column, init=0):
        return NotImplemented

    def num_features(self):
        if len(self.shape) > 1:
            return self.shape[-1]
        else:
            return 1

    def to_df(self):
        return NotImplemented

    def url(self):
        return NotImplemented

    @staticmethod
    def concat(datasets, chunksize:int=0, name:str=None):
        return NotImplemented

    def get_cursor(self, key):
        self.cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        columns = self.columns

        if isinstance(key, tuple):
            _columns = [list(columns.keys())[key[1]]]
            key = key[0]
        else:
            _columns = None

        if isinstance(key, str):
            if _columns is None:
                _columns = [key]
            num_features = len(_columns)
            size = self.shape[0]
            start = 0
        elif isinstance(key, list):
            if _columns is None:
                _columns = key
            num_features = len(_columns)
            size = self.shape[0]
            start = 0
        elif isinstance(key, int):
            if _columns is None:
                _columns = columns.keys()
            num_features = len(_columns)
            size = 1
            start = key
        elif isinstance(key, slice):
            if _columns is None:
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
        elif key is None:
            size = self.shape[0]
            start = 0

        if self.df_dtype is True:
            self.dtype = [(column_name, columns[column_name.lower()]) for column_name in _columns]
        else:
            self.dtype = None

        self.query = "SELECT {columns} FROM {table_name} {order_by} {limit}".format(
            columns=self.format_columns(_columns), table_name=self.table_name,
            order_by=self.order_by_txt,
            limit=self.limit_txt)
        self.cur.execute(self.query)
        self.cur.itersize = self.itersize
        self.cur.scroll(start)
        return size

    def __getitem__(self, key):
        size = self.get_cursor(key)
        it = Iterator(self.cur, dtype=self.dtype)
        it.set_length(size)
        return it.to_memory()

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

    def _build_limit_text(self):
        if self.limit is None:
            self.limit_txt = ""
        else:
            self.limit_txt = "LIMIT {}".format(self.limit)

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

    def reader(self, chunksize:int=0, columns=None, exclude:bool=False, df=True):
        size = self.get_cursor(columns)

        if exclude is True:
            cols = [col for col in self.columns if col not in columns]
        elif exclude is False and columns:
            cols = [col for col in self.columns]
        else:
            cols = None

        #if columns is None:
        it = Iterator(self.cur, chunks_size=chunksize, dtype=None if df is False else self.dtype)
        it.set_length(size)
        #else:
        #    it = Iterator(self[cols], chunks_size=chunksize, dtype=None if df is False else self.dtype)
        return it

    @property
    @cache
    def shape(self):
        cur = self.conn.cursor()
        query = "SELECT COUNT(*) FROM {table_name}".format(
            table_name=self.table_name)
        cur.execute(query)
        size = cur.fetchone()[0]
        return size, len(self.columns)

    def last_id(self):
        cur = self.conn.cursor()
        query = "SELECT last_value FROM {table_name}_id_seq".format(table_name=self.table_name)
        cur.execute(query)
        return cur.fetchone()[0]

    @property
    @cache
    def columns(self):
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s ORDER BY ordinal_position"
        cur.execute(query, {"table_name": self.table_name})
        columns = OrderedDict()
        types = {"text": "|O", "integer": "int", "double precision": "float", "boolean": "bool"}
        if self.only_columns is None:                
            for column in cur.fetchall():
                columns[column[3]] = types.get(column[7], "|O")
            #if exclude_id is True:
            del columns["id"]
        else:
            for column in cur.fetchall():
                if column[3] in self.only_columns:
                    columns[column[3]] = types.get(column[7], "|O")
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

    def insert(self, data, chunks_size=258):
        from psycopg2.extras import execute_values
        from ml.utils.seq import grouper_chunk
        header = self.columns
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
