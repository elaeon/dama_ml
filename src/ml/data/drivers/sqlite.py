from ml.abc.driver import AbsDriver
from ml.data.groups.core import DaGroup
from ml.data.groups.sqlite import Table
from ml.fmtypes import fmtypes_map
from ml.utils.logger import log_config
import numpy as np
import sqlite3

log = log_config(__name__)


class Sqlite(AbsDriver):
    persistent = True
    ext = 'sql'
    data_tag = None
    metadata_tag = None

    def __contains__(self, item):
        return self.exists()

    def enter(self):
        self.conn = sqlite3.connect(self.login.url, check_same_thread=False)
        self.attrs = {}
        if self.mode == "w":
            self.destroy()
        return self

    def exit(self):
        self.conn.close()
        self.attrs = None

    def __enter__(self):
        return self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.exit()

    @property
    def data(self):
        return DaGroup(Table(self.conn, name=self.data_tag))

    def exists(self) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.data_tag, ))
        return True if cur.fetchone() is not None else False

    def destroy(self):
        cur = self.conn.cursor()
        try:
            cur.execute("DROP TABLE {name}".format(name=self.data_tag))
        except sqlite3.ProgrammingError as e:
            log.debug(e)
        self.conn.commit()

    @property
    def dtypes(self) -> np.dtype:
        return Table(self.conn, name=self.data_tag).dtypes

    def set_schema(self, dtypes: np.dtype):
        idx = None
        if not self.exists():
            columns_types = ["id INTEGER PRIMARY KEY"]
            for group, (dtype, _) in dtypes.fields.items():
                fmtype = fmtypes_map[dtype]
                columns_types.append("{col} {type}".format(col=group, type=fmtype.db_type))
            cols = "("+", ".join(columns_types)+")"
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE {name}
                {columns};
            """.format(name=self.data_tag, columns=cols))
            if isinstance(idx, list):
                for index in idx:
                    if isinstance(index, tuple):
                        index_columns = ",".join(index)
                        index_name = "_".join(index)
                    else:
                        index_columns = index
                        index_name = index
                    index_q = "CREATE INDEX {i_name}_{name}_index ON {name} ({i_columns})".format(
                        name=self.data_tag, i_name=index_name, i_columns=index_columns)
                    cur.execute(index_q)
            self.conn.commit()

    def set_data_shape(self, shape):
        pass

    def insert(self, table_name: str, data):
        table = Table(self.conn, table_name)
        table.insert(data)

    def spaces(self) -> list:
        return ["data", "metadata"]
