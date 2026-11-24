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

# B3
def killed_in_action():
    """Return the number of sorcerers killed in action between May 2025 and Oct 2026."""
    return pd.read_sql_query(
        """WITH RECURSIVE
    dedupe AS (
        SELECT DISTINCT mission_id, end_ts, casualties
        FROM missions
        WHERE casualties IS NOT NULL
    ),
    split(mission_id, ts, sorcerer_id, rest) AS (
        SELECT mission_id, end_ts, '', casualties || '|' FROM dedupe
        UNION ALL
        SELECT mission_id, ts,
            substr(rest, 1, instr(rest, '|') - 1),
            substr(rest, instr(rest, '|') + 1)
        FROM split WHERE rest <> ''
    )
    SELECT COUNT(DISTINCT sorcerer_id) AS killed
    FROM split
    WHERE sorcerer_id <> '' AND ts >= '2025-05-01' AND ts < '2026-11-01'""",con,
    ).iloc[0, 0]



if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print(pd.read_sql_query(QUERY, con).to_string(index=False))
    print(f"\nNumber of unique missions in missions: {unique_missions()}")
    print(f"\nNumber of unique wards in missions: {unique_wards_missions()}")
    print(f"\nSorcerers killed in action (May 2025 - Oct 2026): {killed_in_action()}")
