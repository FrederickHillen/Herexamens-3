import sqlite3

import pandas as pd

from infirmary_checks import clean_visits
from load import load_table

con = sqlite3.connect(":memory:")
for table in ("missions", "barrier_sensors", "infirmary_visits"):
    load_table(table).to_sql(table, con, index=False)
# Repaired infirmary visits (discharge UTC->JST from 2026-02-28, severity 1 = critical, test rows removed)
clean_visits().to_sql("infirmary_clean", con, index=False)

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


# C1
def median_recovery_time_by_severity():
    """Return the median recovery time in hours per severity level.

    Recovery time = discharged_at - admitted_at, in hours. A duration needs both ends on the same clock:
    admitted_at is JST, but discharged_at is UTC for visits admitted from 2026-02-28 (9 h too short,
    1086 even negative). So it is computed on clean_visits(), which moves those discharges back to JST,
    reverses severity to the dictionary's 1 = critical ... 5 = minor and drops the TST-00 test records.
    """
    sql = """
    WITH recovery AS (
        SELECT severity,
               (julianday(discharged_at) - julianday(admitted_at)) * 24 AS hours
        FROM infirmary_clean
        WHERE admitted_at IS NOT NULL AND discharged_at IS NOT NULL
    ),
    ordered AS (
        SELECT severity, hours,
               ROW_NUMBER() OVER (PARTITION BY severity ORDER BY hours) AS rn,
               COUNT(*) OVER (PARTITION BY severity) AS n
        FROM recovery
    )
    SELECT severity, ROUND(AVG(hours), 2) AS median_recovery_hours
    FROM ordered
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY severity
    ORDER BY severity
    """
    return pd.read_sql_query(sql, con)

def post_mortem():
    sql = """WITH RECURSIVE
    m AS (SELECT DISTINCT * FROM missions),                       -- drop exact duplicate rows
    cas AS (SELECT mission_id, end_ts AS died_at, casualties AS sorcerer_id
            FROM m WHERE casualties IS NOT NULL),
    sp(mission_id, sorcerer_id, rest) AS (                        -- split 'S-0001|S-0002'
    SELECT mission_id, '', sorcerer_ids || '|' FROM m
    UNION ALL SELECT mission_id, substr(rest,1,instr(rest,'|')-1), substr(rest,instr(rest,'|')+1)
    FROM sp WHERE rest <> ''),
    ms AS (SELECT mission_id, sorcerer_id FROM sp WHERE sorcerer_id <> '')
    SELECT s.sorcerer_id, s.name, s.status, c.mission_id AS casualty_mission, c.died_at,
        COUNT(m2.mission_id) AS missions_after_death, MAX(m2.start_ts) AS last_mission
    FROM cas c
    JOIN sorcerers s USING (sorcerer_id)
    LEFT JOIN ms ON ms.sorcerer_id = c.sorcerer_id
    LEFT JOIN m m2 ON m2.mission_id = ms.mission_id AND m2.start_ts > c.died_at
    WHERE s.status <> 'deceased'
    GROUP BY s.sorcerer_id ORDER BY c.died_at"""
    return pd.read_sql_query(sql, con)

if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print(pd.read_sql_query(QUERY, con).to_string(index=False))
    print(f"\nNumber of unique missions in missions: {unique_missions()}")
    print(f"\nNumber of unique wards in missions: {unique_wards_missions()}")
    print(f"\nMedian residual cursed energy in barrier_sensors: {median_residual_cursed_energy()}")
    print(f"\nSorcerers killed in action (May 2025 - Oct 2026): {killed_in_action()}")
    print(f"\nMedian recovery time (hours) by severity: {median_recovery_time_by_severity().to_string(index=False)}")
