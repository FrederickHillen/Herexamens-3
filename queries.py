import sqlite3

import pandas as pd

from load import load_table

con = sqlite3.connect(":memory:")
for table in ("missions", "barrier_sensors"):
    load_table(table).to_sql(table, con, index=False)

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

#B2
def median_residual_cursed_energy():
    """Return the median residual cursed energy in the barrier_sensors table."""
    sql = """
    WITH ordered AS (
        SELECT residual_ce,
               ROW_NUMBER() OVER (ORDER BY residual_ce) AS rn,
               COUNT(*) OVER () AS n
        FROM barrier_sensors
        WHERE residual_ce IS NOT NULL
    )
    SELECT AVG(residual_ce) AS median_residual_cursed_energy
    FROM ordered
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    """
    return pd.read_sql_query(sql, con).iloc[0, 0]

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print(pd.read_sql_query(QUERY, con).to_string(index=False))
    print(f"\nNumber of unique missions in missions: {unique_missions()}")
    print(f"\nNumber of unique wards in missions: {unique_wards_missions()}")
    print(f"\nMedian residual cursed energy in barrier_sensors: {median_residual_cursed_energy()}")
