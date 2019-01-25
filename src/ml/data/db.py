import psycopg2
import uuid
import numpy as np

from tqdm import tqdm
from ml.utils.logger import log_config
from collections import OrderedDict
from psycopg2.extras import execute_values
from ml.fmtypes import fmtypes_map
from ml.utils.basic import Shape
from ml.abc.driver import AbsDriver
from ml.utils.numeric_functions import max_dtype, all_int
from ml.data.it import Iterator, BatchIterator
from ml.utils.decorators import cache
from ml.abc.group import AbsGroup

log = log_config(__name__)


class Postgres(AbsDriver):
    persistent = True
    ext = 'sql'

    def __contains__(self, item):
        return self.exists(item)

    def enter(self, url):
        self.conn = psycopg2.connect(
            "dbname={db_name} user={username}".format(db_name=self.login.resource, username=self.login.username))
        self.conn.autocommit = False
        self.attrs = {}
        return self

    def exit(self):
        self.conn.close()
        self.attrs = None

    def __enter__(self):
        return self.enter(None)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.exit()

    def __getitem__(self, item):
        if isinstance(item, str):
            return Table(self.conn, name=item)

    def exists(self, scope) -> bool:
        cur = self.conn.cursor()
        cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=scope))
        return True if cur.fetchone()[0] else False

    def destroy(self, scope):
        cur = self.conn.cursor()
        try:
            cur.execute("DROP TABLE {name}".format(name=scope))
        except psycopg2.ProgrammingError as e:
            log.debug(e)
        self.conn.commit()

    def dtypes(self, name) -> list:
        return self[name].dtypes

    def set_schema(self, name, dtypes):
        idx = None
        if not self.exists(name):
            columns_types = ["id serial PRIMARY KEY"]
            for col, dtype in dtypes:
                fmtype = fmtypes_map[dtype]
                columns_types.append("{col} {type}".format(col=col, type=fmtype.db_type))
            cols = "("+", ".join(columns_types)+")"
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE {name}
                {columns};
            """.format(name=name, columns=cols))
            if isinstance(idx, list):
                for index in idx:
                    if isinstance(index, tuple):
                        index_columns = ",".join(index)
                        index_name = "_".join(index)
                    else:
                        index_columns = index
                        index_name = index
                    index_q = "CREATE INDEX {i_name}_{name}_index ON {name} ({i_columns})".format(
                        name=name, i_name=index_name, i_columns=index_columns)
                    cur.execute(index_q)
            self.conn.commit()

    def set_data_shape(self, shape):
        pass

    def insert(self, table_name: str, data):
        table = Table(self.conn, table_name)
        table.insert(data)


class Table(AbsGroup):
    def __init__(self, conn, name=None, query_parts=None):
        super(Table, self).__init__(conn, name=name)
        if query_parts is None:
            self.query_parts = {"columns": None, "slice": None}
        else:
            self.query_parts = query_parts

    def __getitem__(self, item):
        query_parts = self.query_parts.copy()
        if isinstance(item, str):
            query_parts["columns"] = [item]
        elif isinstance(item, list) or isinstance(item, tuple):
            if all_int(item):
                query_parts["slice"] = [slice(index, index + 1) for index in item]
            else:
                query_parts["columns"] = item
        elif isinstance(item, int):
            query_parts["slice"] = slice(item, item + 1)
        elif isinstance(item, slice):
            query_parts["slice"] = item
        return Table(self.conn, name=self.name, query_parts=query_parts)

    def __setitem__(self, item, value):
        if isinstance(item, slice):
            last_id = self.last_id()
            if last_id < item.stop:
                self.insert(value, batch_size=abs(item.stop - item.start))
            else:
                self.update(value)

    def __iter__(self):
        pass

    def insert(self, data, batch_size=258):
        if not isinstance(data, BatchIterator):
            data = Iterator(data).batchs(batch_size=batch_size, batch_type="array")
        columns = "(" + ", ".join([name for name, _ in self.dtypes]) + ")"
        insert_str = "INSERT INTO {name} {columns} VALUES".format(
            name=self.name, columns=columns)
        insert = insert_str + " %s"
        cur = self.conn.cursor()
        for row in tqdm(data, total=len(data)):
            execute_values(cur, insert, row, page_size=data.batch_size)
        self.conn.commit()

    def update(self, value):
        raise NotImplementedError

    def compute(self, chunksize=(258,)):
        slice_item, _ = self.build_limit_info()
        query = self.build_query()
        cur = self.conn.cursor(uuid.uuid4().hex, scrollable=False, withhold=False)
        cur.execute(query)
        cur.itersize = chunksize[0]
        cur.scroll(slice_item.start)
        array = np.empty(self.shape, self.dtype)
        if len(self.shape.groups()) == 1:
            for i, row in enumerate(cur):
                array[i] = row[0]
        else:
            array[:] = cur.fetchall()
        self.conn.commit()
        return array

    @property
    @cache
    def shape(self) -> Shape:
        cur = self.conn.cursor()
        slice_item, limit_txt = self.build_limit_info()
        if limit_txt == "":
            query = "SELECT COUNT(*) FROM {table_name}".format(table_name=self.name)
            cur.execute(query)
            length = cur.fetchone()[0]
        else:
            query = "SELECT COUNT(*) FROM {table_name} WHERE id > {start} and id <= {stop}".format(
                start=slice_item.start, stop=slice_item.stop, table_name=self.name)
            cur.execute(query)
            length = cur.fetchone()[0]
        shape = dict([(group, (length,)) for group, _ in self.dtypes])
        return Shape(shape)

    @property
    @cache
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

    def format_columns(self):
        columns = self.query_parts["columns"]
        if columns is None:
            columns = [column for column, _ in self.dtypes]
        return ",".join(columns)

    def build_limit_info(self) -> tuple:
        if isinstance(self.query_parts["slice"], list):
            index = [index.start - 1 for index in self.query_parts["slice"]]
            min_elem = min(index)
            stop = len(index)
            return slice(min_elem, stop), "LIMIT {}".format(stop)

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
        if isinstance(self.query_parts["slice"], list):
            id_list = [index.start + 1 for index in self.query_parts["slice"]]
            query = "SELECT {columns} FROM {table_name} WHERE ID IN ({id_list}) ORDER BY {order_by}".format(
                columns=self.format_columns(), table_name=self.name, order_by="id",
                id_list=",".join(map(str, id_list)))
        else:
            slice_item, limit_txt = self.build_limit_info()
            query = "SELECT {columns} FROM {table_name} ORDER BY {order_by} {limit}".format(
                columns=self.format_columns(), table_name=self.name, order_by="id",
                limit=limit_txt)
        return query
