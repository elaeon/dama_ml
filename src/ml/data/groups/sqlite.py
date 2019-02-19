from ml.abc.group import AbsGroup
from ml.utils.core import Shape
import numpy as np
from collections import OrderedDict
from ml.utils.decorators import cache
from ml.data.it import Iterator, BatchIterator


class Table(AbsGroup):
    inblock = True

    def __init__(self, conn, name=None, query_parts=None):
        super(Table, self).__init__(conn)
        self.name = name
        if query_parts is None:
            self.query_parts = {"columns": None, "slice": None}
        else:
            self.query_parts = query_parts

    def __getitem__(self, item):
        query_parts = self.query_parts.copy()
        if isinstance(item, str):
            query_parts["columns"] = [item]
            return Table(self.conn, name=self.name, query_parts=query_parts)
        elif isinstance(item, list) or isinstance(item, tuple):
            it = Iterator(item)
            if it.type_elem == int:
                query_parts["slice"] = [slice(index, index + 1) for index in item]
            elif it.type_elem == slice:
                query_parts["slice"] = item
            elif it.type_elem == str:
                query_parts["columns"] = item
            dtype = self.attrs.get("dtype", None)
            return Table(self.conn, name=self.name, query_parts=query_parts).to_ndarray(dtype=dtype)
        elif isinstance(item, int):
            query_parts["slice"] = slice(item, item + 1)
            dtype = self.attrs.get("dtype", None)
            return Table(self.conn, name=self.name, query_parts=query_parts).to_ndarray(dtype=dtype)
        elif isinstance(item, slice):
            query_parts["slice"] = item
            dtype = self.attrs.get("dtype", None)
            return Table(self.conn, name=self.name, query_parts=query_parts).to_ndarray(dtype=dtype)

    def __setitem__(self, item, value):
        if hasattr(value, 'batch'):
            value = value.batch

        if isinstance(item, tuple):
            if len(item) == 1:
                stop = item[0].stop
                start = item[0].start
            else:
                raise NotImplementedError
            batch_size = abs(stop - start)
        elif isinstance(item, slice):
            stop = item.stop
            start = item.start
            batch_size = abs(stop - start)
        elif isinstance(item, int):
            stop = item + 1
            if hasattr(value, '__len__'):
                batch_size = len(value)
            else:
                batch_size = 1
        else:
            raise NotImplementedError

        last_id = self.last_id()
        if last_id < stop:
            self.insert(value, batch_size=batch_size)
        else:
            self.update(value, item)

    def __iter__(self):
        pass

    def get_group(self, group):
        return self[group]

    def get_conn(self, group):
        return self[group]

    def insert(self, data, batch_size=258):
        if not isinstance(data, BatchIterator):
            data = Iterator(data, dtypes=self.dtypes)
            chunks = data.shape.to_chunks(batch_size)
            data = data.batchs(chunks=chunks)

        columns = "(" + ", ".join(self.groups) + ")"
        values = "(" + "?,".join(("" for _ in self.groups)) + "?)"
        insert_str = "INSERT INTO {name} {columns} VALUES {values}".format(
            name=self.name, columns=columns, values=values)
        cur = self.conn.cursor()
        num_groups = len(data.groups)
        for row in data:
            shape = row.batch.shape.to_tuple()
            if len(shape) == 1 and num_groups > 1:
                value = row.batch.to_df.values  # .to_ndarray().reshape(1, -1)
            elif len(shape) == 1 and num_groups == 1:
                value = row.batch.to_df().values  # .to_ndarray().reshape(-1, 1)
            else:
                value = row.batch.to_ndarray(object)
            cur.executemany(insert_str, value)
        self.conn.commit()
        cur.close()

    def update(self, value, item):
        if isinstance(item, int):
            columns_values = [[self.groups[0], value]]
            columns_values = ["{col}={val}".format(col=col, val=val) for col, val in columns_values]
            query = "UPDATE {name} SET {columns_val} WHERE ID = {id}".format(
                name=self.name, columns_val=",".join(columns_values), id=item+1
            )
            cur = self.conn.cursor()
            cur.execute(query)
            self.conn.commit()
            cur.close()
        else:
            raise NotImplementedError

    def to_ndarray(self, dtype: np.dtype = None, chunksize=(258,)) -> np.ndarray:
        if self.dtype is None:
            return np.asarray([])

        query, one_row = self.build_query()
        cur = self.conn.cursor()
        cur.execute(query)
        array = np.empty(self.shape, dtype=self.dtype)
        if len(self.groups) == 1:
            for i, row in enumerate(cur):
                array[i] = row[0]
        else:
            array[:] = cur.fetchall()
        self.conn.commit()
        cur.close()
        if dtype is not None and self.dtype != dtype:
            return array.astype(dtype)
        else:
            return array

    def to_df(self):
        pass

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
            query = "SELECT Count(*) FROM (SELECT id FROM {table_name} LIMIT {limit} OFFSET {start})".format(
                table_name=self.name, start=slice_item.start, limit=(abs(slice_item.stop - slice_item.start)))
            cur.execute(query)
            length = cur.fetchone()[0]
        shape = OrderedDict([(group, (length,)) for group in self.groups])
        cur.close()
        return Shape(shape)

    @property
    @cache
    def dtypes(self) -> np.dtype:
        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info('{}')".format(self.name))
        dtypes = OrderedDict()
        types = {"text": np.dtype("object"), "integer": np.dtype("int"),
                 "float": np.dtype("float"), "boolean": np.dtype("bool"),
                 "timestamp": np.dtype('datetime64[ns]')}

        if self.query_parts["columns"] is not None:
            for column in cur.fetchall():
                if column[1] in self.query_parts["columns"]:
                    dtypes[column[1]] = types.get(column[2].lower(), np.dtype("object"))
        else:
            for column in cur.fetchall():
                dtypes[column[1]] = types.get(column[2].lower(), np.dtype("object"))

        if "id" in dtypes:
            del dtypes["id"]

        cur.close()
        if len(dtypes) > 0:
            return np.dtype(list(dtypes.items()))

    def last_id(self):
        cur = self.conn.cursor()
        cur.execute("SELECT max(id) FROM {}".format(self.name))
        id_ = cur.fetchone()[0]
        cur.close()
        return id_

    def format_columns(self):
        columns = self.query_parts["columns"]
        if columns is None:
            columns = [column for column, _ in self.dtypes]
        return ",".join(columns)

    def build_limit_info(self) -> tuple:
        if isinstance(self.query_parts["slice"], list):
            index_start = [index.start for index in self.query_parts["slice"]]
            index_stop = [index.stop for index in self.query_parts["slice"]]
            min_elem = min(index_start)
            max_elem = max(index_stop)
            return slice(min_elem, max_elem), "LIMIT {}".format(max_elem)
        elif isinstance(self.query_parts["slice"], tuple):
            item = self.query_parts["slice"][0]
        else:
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

    def build_query(self) -> tuple:
        if isinstance(self.query_parts["slice"], list):
            id_list = [index.start + 1 for index in self.query_parts["slice"]]
            query = "SELECT {columns} FROM {table_name} WHERE ID IN ({id_list}) ORDER BY {order_by}".format(
                columns=self.format_columns(), table_name=self.name, order_by="id",
                id_list=",".join(map(str, id_list)))
            one_row = True
        else:
            slice_item, limit_txt = self.build_limit_info()
            if limit_txt == "":
                query = "SELECT {columns} FROM {table_name} ORDER BY {order_by}".format(
                    columns=self.format_columns(), table_name=self.name, order_by="id")
            else:
                query = "SELECT {columns} FROM {table_name} ORDER BY {order_by} LIMIT {limit} OFFSET {start}".format(
                    columns=self.format_columns(), table_name=self.name, order_by="id", start=slice_item.start,
                    limit=(abs(slice_item.stop - slice_item.start)))
            one_row = False
        return query, one_row
