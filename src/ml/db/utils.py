import psycopg2

def build_schema(table_name, columns_name, fmtypes, id_name):
    conn = psycopg2.connect("dbname=ml user=alejandro")
    cur = conn.cursor()
    cur.execute("select exists(select relname from pg_class where relname='{name}')".format(name=table_name))
    exists = cur.fetchone()[0]
    if not exists:
        columns_types = ["id serial PRIMARY KEY"]
        for col, fmtype in zip(columns_name, fmtypes):
            columns_types.append("{col} {type}".format(col=col, type=fmtype.db_type))
        cols = "("+", ".join(columns_types)+")"
        index = "CREATE INDEX {id_name}_index ON {name} ({id_name})".format(name=table_name, id_name=id_name)
        
        cur.execute("""
            CREATE TABLE {name}
            {columns};
        """.format(name=table_name, columns=cols))
        cur.execute(index)
    else:
        cur.execute("DROP TABLE {name};".format(name=table_name))
    conn.commit()
    cur.close()
    conn.close()
    print("CLOSE")


def insert_rows(csv_reader, table_name, header):
    columns = "("+", ".join(header)+")"
    insert_str = "INSERT INTO {name} {columns} VALUES".format(name=table_name, columns=columns)
    values_str = "("+", ".join(["%s" for _ in range(len(header))])+")"
    insert = insert_str+" "+values_str

    conn = psycopg2.connect("dbname=ml user=alejandro")
    cur = conn.cursor()
    for row in csv_reader:
        print(row)
        cur.execute(insert, row)
        break
    conn.commit()
    cur.close()
    conn.close()
