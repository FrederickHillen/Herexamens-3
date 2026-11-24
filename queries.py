import sqlite3

import pandas as pd

from load import load_table

# Only the missions table is loaded into SQLite.
con = sqlite3.connect(":memory:")
load_table("missions").to_sql("missions", con, index=False)

QUERY = """
SELECT * FROM missions LIMIT 5
"""  # replace with your query

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print(pd.read_sql_query(QUERY, con).to_string(index=False))
