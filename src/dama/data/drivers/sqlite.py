from dama.abc.driver import AbsDriver
from dama.data.groups.sqlite import Table
from dama.fmtypes import fmtypes_map
from dama.utils.logger import log_config
import numpy as np
import sqlite3

log = log_config(__name__)


class Sqlite(AbsDriver):
    persistent = True
    ext = 'sqlite3'
    data_tag = None
    metadata_tag = None

    def __contains__(self, item):
        return self.exists()

    def open(self):
        self.conn = sqlite3.connect(self.url, check_same_thread=False)
        self.data_tag = self.login.table
        self.attrs = {}
        if self.mode == "w":
            self.destroy()
        return self

    def close(self):
        self.conn.close()
        self.attrs = None

    @property
    def absgroup(self):
        return Table(self.conn, name=self.data_tag)

    def exists(self) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.data_tag, ))
        result = cur.fetchone()
        return result is not None

    def destroy(self):
        cur = self.conn.cursor()
        try:
            cur.execute("DROP TABLE {name}".format(name=self.data_tag))
        except sqlite3.ProgrammingError as e:
            log.debug(e)
        except sqlite3.OperationalError as e:
            log.error(e)
        self.conn.commit()

    @property
    def dtypes(self) -> np.dtype:
        return self.absgroup.dtypes

    @property
    def groups(self) -> tuple:
        return self.absgroup.groups

    def set_schema(self, dtypes: np.dtype, idx: list = None, unique_key: list = None):
        if not self.exists():
            columns_types = ["id INTEGER PRIMARY KEY"]
            if unique_key is not None:
                one_col_unique_key = [column for column in unique_key if isinstance(column, str)]
                more_col_unique_key = [columns for columns in unique_key if isinstance(columns, list)]
            else:
                one_col_unique_key = []
                more_col_unique_key = []
            for group, (dtype, _) in dtypes.fields.items():
                fmtype = fmtypes_map[dtype]
                if group in one_col_unique_key:
                    columns_types.append("{col} {type} UNIQUE".format(col=group, type=fmtype.db_type))
                else:
                    columns_types.append("{col} {type}".format(col=group, type=fmtype.db_type))
            if len(more_col_unique_key) > 0:
                for key in more_col_unique_key:
                    columns_types.append("unique ({})".format(",".join(key)))
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
            cur.close()

    def set_data_shape(self, shape):
        pass

    def insert(self, data):
        table = Table(self.conn, self.data_tag)
        table.insert(data)

    def spaces(self) -> list:
        return ["data", "metadata"]
