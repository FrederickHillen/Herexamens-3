import sqlite3

import pandas as pd

from load import load_table

# Only the missions table is loaded into SQLite.
con = sqlite3.connect(":memory:")
load_table("missions").to_sql("missions", con, index=False)

QUERY = """
SELECT * FROM missions LIMIT 5
"""  # replace with your query

# A2
def unique_missions():
    """Return the number of unique missions in the missions table."""
    return pd.read_sql_query("SELECT COUNT(DISTINCT mission_id) AS unique_missions FROM missions", con).iloc[0, 0]

#B1
def unique_wards_missions():
    """Return the number of unique wards in the missions table."""
    return pd.read_sql_query("SELECT COUNT(DISTINCT ward) AS unique_wards FROM missions", con).iloc[0, 0]


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print(pd.read_sql_query(QUERY, con).to_string(index=False))
    print(f"\nNumber of unique missions in missions: {unique_missions()}")
    print(f"\nNumber of unique wards in missions: {unique_wards_missions()}")
