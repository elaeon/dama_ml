import psycopg2
import uuid
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
from psycopg2.extras import execute_values
from ml.fmtypes import fmtypes_map
from ml.utils.basic import Shape
from ml.abc.driver import AbsDriver
from ml.utils.numeric_functions import max_dtype


class Schema(AbsDriver):
    persistent = True
    ext = 'sql'

    def __contains__(self, item):
        return self.exists(item)

    def enter(self, url):
        self.__enter__()

    def exit(self):
        self.__exit__()

    def __enter__(self):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.login.resource, username=self.login.username))
        self.conn.autocommit = False
        self.attrs = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        self.attrs = None

    def __getitem__(self, item):
        if isinstance(item, str):
            return Table(item, self.conn)

    def exists(self, table_name) -> bool:
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=table_name))
        return True if cur.fetchone()[0] else False

    def build(self, table_name, columns, indexes=None):
        if not self.exists(table_name):
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
            self.conn.commit()

    def destroy(self, table_name):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE {name}".format(name=table_name))
        self.conn.commit()

    def dtypes(self, table_name) -> list:
        return self[table_name].dtypes()

    def insert(self, table_name: str, data):
        header = self.dtypes(table_name)
        columns = "(" + ", ".join([name for name, _ in header]) + ")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(
            name=table_name, columns=columns)
        insert = insert_str + " %s"
        cur = self.conn.cursor()
        for row in tqdm(data, total=data.num_splits()):
            execute_values(cur, insert, row, page_size=data.batch_size)
        self.conn.commit()

    def require_group(self, *args, **kwargs):
        return self.f.require_group(*args, **kwargs)

    def require_dataset(self, group: str, name: str, shape: tuple, dtype: np.dtype) -> None:
        self.chunk_size = 1000
        self.f[group].require_dataset(name, shape, dtype=dtype, chunks=True,
                                      exact=True,
                                      **self.compressor_params)

    def auto_dtype(self, dtype: np.dtype):
        return dtype


class Table(object):
    def __init__(self, name, conn, query_parts=None):
        self.conn = conn
        self.name = name
        if query_parts is None:
            self.query_parts = {"columns": None, "slice": None}
        else:
            self.query_parts = query_parts

    def __getitem__(self, item):
        if isinstance(item, str):
            self.query_parts["columns"] = [item]
        elif isinstance(item, list):
            self.query_parts["columns"] = item
        elif isinstance(item, int):
            self.query_parts["slice"] = slice(item, item + 1)
        elif isinstance(item, slice):
            self.query_parts["slice"] = item
        return self

    def compute(self):
        slice_item, _ = self.build_limit_info()
        query = self.build_query()
        cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        cur.execute(query)
        cur.itersize = 1000  # chunksize
        cur.scroll(slice_item.start)
        print(self.shape)
        array = np.empty(self.shape, max_dtype(self.dtypes()))
        print(cur.fetchall())
        print(query)
        #array[:] = cur.fetchall()
        self.conn.commit()
        return array

    @property
    def shape(self) -> Shape:
        cur = self.conn.cursor()
        slice_item, limit_txt = self.build_limit_info()
        if limit_txt == "":
            query = "SELECT COUNT(*) FROM {table_name}".format(table_name=self.name)
            cur.execute(query)
            length = cur.fetchone()[0]
        else:
            length = abs(slice_item.stop - slice_item.start)
        return Shape({self.name: (length, len(self.dtypes()))})

    def dtypes(self) -> list:
        cur = self.conn.cursor()
        query = "SELECT * FROM information_schema.columns WHERE table_name=%(table_name)s ORDER BY ordinal_position"
        cur.execute(query, {"table_name": self.name})
        dtypes = OrderedDict()
        types = {"text": np.dtype("object"), "integer": np.dtype("int"),
                 "double precision": np.dtype("float"), "boolean": np.dtype("bool"),
                 "timestamp without time zone": np.dtype('datetime64[ns]')}

        if self.query_parts["columns"] is not None:
            for column in cur.fetchall():
                if column[3] in self.query_parts["columns"]:
                    dtypes[column[3]] = types.get(column[7], np.dtype("object"))
        else:
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

    @staticmethod
    def format_columns(columns):
        if columns is None:
            return "*"
        else:
            return ",".join(columns)

    def build_limit_info(self) -> tuple:
        item = self.query_parts["slice"]
        if item is None:
            start = 0
            stop = None
            limit_txt = ""
        else:
            if item.start is None:
                start = 0
            else:
                start = item.start

            if item.stop is None:
                limit_txt = ""
                stop = None
            else:
                limit_txt = "LIMIT {}".format(item.stop)
                stop = item.stop

        return slice(start, stop), limit_txt

    def build_query(self) -> str:
        slice_item, limit_txt = self.build_limit_info()
        query = "SELECT {columns} FROM {table_name} ORDER BY {order_by} {limit}".format(
            columns=Table.format_columns(self.query_parts["columns"]), table_name=self.name, order_by="id",
            limit=limit_txt)
        return query
