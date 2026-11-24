"""Check and repair infirmary_visits.csv.

check_infirmary()            -> DataFrame of (visit_id, issue), one row per problem found
fix_infirmary(out_file)      -> applies the safe fixes and writes the result to out_file
"""
import sqlite3
from datetime import timedelta

import pandas as pd

from holders import reused_ids
from load import load_table

# Discharge times switched from JST to UTC for visits admitted from about here
# (first visit that ends before it starts: V-004514, admitted 2026-02-28 19:14).
UTC_DISCHARGE_FROM = pd.Timestamp("2026-02-28 15:00")


PREP_SQL = """
CREATE TABLE m AS SELECT DISTINCT * FROM missions;  -- exact duplicate rows removed
CREATE INDEX m_id ON m(mission_id);
CREATE TABLE mission_sorcerers AS
WITH RECURSIVE sp(mission_id, sorcerer_id, rest) AS (
  SELECT mission_id, '', sorcerer_ids || '|' FROM m
  UNION ALL SELECT mission_id, substr(rest,1,instr(rest,'|')-1), substr(rest,instr(rest,'|')+1) FROM sp WHERE rest<>'')
SELECT mission_id, sorcerer_id FROM sp WHERE sorcerer_id<>'';
CREATE INDEX ms_s ON mission_sorcerers(sorcerer_id); CREATE INDEX ms_m ON mission_sorcerers(mission_id, sorcerer_id);
"""

CHECK_SQL_TEMPLATE = """
WITH
v AS (  -- infirmary visits with JST converted to UTC
    -- discharge times of later visits are already UTC, so they get no -9h (see UTC_DISCHARGE_FROM)
    SELECT *, datetime(admitted_at || ':00', '-9 hours') AS adm_utc,
              CASE WHEN admitted_at >= '__CUTOFF__' THEN datetime(discharged_at || ':00')
                   ELSE datetime(discharged_at || ':00', '-9 hours') END AS dis_utc,
              admitted_at >= '__CUTOFF__' AS dis_is_utc
    FROM infirmary_visits
),
issues AS (
    -- 1. visit points at a mission that does not exist (or no mission at all)
    SELECT v.visit_id, 'mission_missing' AS issue FROM v
    LEFT JOIN m USING (mission_id) WHERE m.mission_id IS NULL
    UNION ALL
    -- 2. patient was not assigned to that mission
    SELECT v.visit_id, 'patient_not_in_mission' FROM v JOIN m USING (mission_id)
    WHERE NOT EXISTS (SELECT 1 FROM mission_sorcerers ms WHERE ms.mission_id = v.mission_id AND ms.sorcerer_id = v.sorcerer_id)
    UNION ALL
    -- 3. admitted before the mission started (should be admitted after it ended)
    SELECT v.visit_id, 'admitted_before_mission_end' FROM v JOIN m USING (mission_id)
    WHERE v.adm_utc < m.end_ts
    UNION ALL
    -- 4. discharge before admission (after the timezone correction), and uncorrected UTC discharges
    SELECT visit_id, 'discharge_before_admit' FROM v WHERE dis_utc < adm_utc
    UNION ALL
    SELECT visit_id, 'discharge_stamped_utc' FROM v WHERE dis_is_utc
    UNION ALL
    -- 5. zero-length stay
    SELECT visit_id, 'zero_length_stay' FROM v WHERE dis_utc = adm_utc
    UNION ALL
    -- 6. patient started another mission while still admitted
    SELECT DISTINCT v.visit_id, 'mission_during_stay' FROM v
    JOIN mission_sorcerers ms ON ms.sorcerer_id = v.sorcerer_id JOIN m m2 ON m2.mission_id = ms.mission_id
    WHERE m2.mission_id <> v.mission_id AND m2.start_ts > v.adm_utc AND m2.start_ts < v.dis_utc
    UNION ALL
    -- 7. patient was already admitted for another visit that overlaps this one
    SELECT DISTINCT a.visit_id, 'overlapping_visit' FROM v a JOIN v b
      ON a.sorcerer_id = b.sorcerer_id AND a.visit_id <> b.visit_id
     AND a.adm_utc < b.dis_utc AND b.adm_utc < a.dis_utc
    UNION ALL
    -- 8. patient was on another mission at the time of admission
    SELECT DISTINCT v.visit_id, 'on_other_mission_at_admission' FROM v
    JOIN mission_sorcerers ms ON ms.sorcerer_id = v.sorcerer_id JOIN m m2 ON m2.mission_id = ms.mission_id
    WHERE m2.mission_id <> v.mission_id AND v.adm_utc BETWEEN m2.start_ts AND m2.end_ts
    UNION ALL
    -- 9. injury code format (CCC-NN, upper case)
    SELECT visit_id, 'injury_code_format' FROM v WHERE injury_code NOT GLOB '[A-Z][A-Z][A-Z]-[0-9][0-9]'
    UNION ALL
    -- 10. injury code not in known list (TST-00 = test record) or not matching severity
    SELECT visit_id, 'injury_code_unknown' FROM v
    WHERE upper(substr(injury_code,1,3) || '-' || substr(injury_code,-2)) NOT IN
      ('ABR-01','BRU-01','SPR-01','BRN-02','FRC-02','LAC-02','CEP-03','FRC-03','LAC-03',
       'BRN-04','CEP-04','PEN-04','CEX-05','CRU-05','PEN-05')
    UNION ALL
    SELECT visit_id, 'injury_code_severity_mismatch' FROM v
    WHERE CAST(substr(injury_code,-1) AS INTEGER) <> severity
    UNION ALL
    -- 11. severity outside 1..5
    SELECT visit_id, 'severity_out_of_range' FROM v WHERE severity NOT BETWEEN 1 AND 5
    UNION ALL
    -- 12. unknown sorcerer, admitted before joining, or admitted after death.
    --     IDs were reused (see holders.py): a visit before join_date is fine if it belongs to the first
    --     holder (before their death); a visit after the casualty is fine if it belongs to the new holder.
    SELECT v.visit_id, 'unknown_sorcerer' FROM v
    LEFT JOIN sorcerers s USING (sorcerer_id) WHERE s.sorcerer_id IS NULL
    UNION ALL
    SELECT v.visit_id, 'admitted_before_join_date' FROM v JOIN sorcerers s USING (sorcerer_id)
    WHERE date(v.adm_utc) < s.join_date
      AND NOT EXISTS (SELECT 1 FROM reused_ids r WHERE r.sorcerer_id = v.sorcerer_id AND v.adm_utc <= r.died_at)
    UNION ALL
    SELECT v.visit_id, 'admitted_after_death' FROM v JOIN (SELECT * FROM m WHERE casualties IS NOT NULL) m
      ON '|' || m.casualties || '|' LIKE '%|' || v.sorcerer_id || '|%'
    WHERE m.start_ts < v.adm_utc AND m.mission_id <> v.mission_id
      AND NOT EXISTS (SELECT 1 FROM reused_ids r WHERE r.sorcerer_id = v.sorcerer_id AND v.adm_utc >= r.rejoined_at)
    UNION ALL
    -- 13. duplicate visit_id, and duplicate visits (same patient + mission)
    SELECT visit_id, 'duplicate_visit_id' FROM v WHERE visit_id IN
      (SELECT visit_id FROM v GROUP BY visit_id HAVING COUNT(*) > 1)
    UNION ALL
    SELECT visit_id, 'duplicate_patient_mission' FROM v
    WHERE mission_id IS NOT NULL AND (sorcerer_id, mission_id) IN
      (SELECT sorcerer_id, mission_id FROM v GROUP BY 1, 2 HAVING COUNT(*) > 1)
    UNION ALL
    -- 14. missing values / bad ID formats
    SELECT visit_id, 'null_or_bad_format' FROM v
    WHERE visit_id NOT GLOB 'V-[0-9][0-9][0-9][0-9][0-9][0-9]'
       OR sorcerer_id NOT GLOB 'S-[0-9][0-9][0-9][0-9]'
       OR admitted_at IS NULL OR discharged_at IS NULL OR admitted_from IS NULL
)
SELECT visit_id, issue FROM issues ORDER BY visit_id, issue
"""
CHECK_SQL = CHECK_SQL_TEMPLATE.replace("__CUTOFF__", UTC_DISCHARGE_FROM.strftime("%Y-%m-%d %H:%M"))


def _connect():
    con = sqlite3.connect(":memory:")
    for t in ("missions", "sorcerers", "infirmary_visits"):
        load_table(t).to_sql(t, con, index=False)
    r = reused_ids()
    for c in ("died_at", "rejoined_at"):
        r[c] = r[c].dt.strftime("%Y-%m-%d %H:%M:%S")
    r.to_sql("reused_ids", con, index=False)
    con.executescript(PREP_SQL)
    return con


def check_infirmary(summary=False):
    """Run all checks on infirmary_visits.csv.

    Returns a DataFrame with one row per (visit_id, issue).
    With summary=True returns the number of visits per issue instead.
    """
    issues = pd.read_sql_query(CHECK_SQL, _connect())
    if summary:
        return issues.groupby("issue").size().sort_values(ascending=False).rename("n_visits").reset_index()
    return issues


def fix_infirmary(out_file="infirmary_visits_fixed.csv"):
    """Write clean_visits() to out_file (the original is not touched)."""
    df = clean_visits()
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out_file} (discharge times shifted, test rows removed, severity reversed)")
    return df


def clean_visits():
    """Return infirmary_visits with the safe fixes applied (timestamps stay JST).

    Fixes:
      * injury_code: upper-case and add the hyphen (abr01 -> ABR-01)
      * discharged_at: from visits admitted on/after UTC_DISCHARGE_FROM (end of Feb 2026) the
        discharge time was stamped in UTC instead of JST -> add 9 hours to ALL of them
        (not only the ones that ended up before admission)
      * severity: the file has 1 = minor ... 5 = critical, the opposite of the data dictionary
        (1 = critical ... 5 = minor), so it is reversed (severity = 6 - severity).
        Evidence: median stay grows with the raw value (about 3 h at 1, 240 h at 5) and
        the injury code suffix follows it (ABR-01 abrasion = 1, CRU-05 crush = 5).
      * test records (TST-00, no mission) are removed
    Timestamps stay in JST, as in the original file.
    NB: the injury_code_severity_mismatch check compares the code suffix with the RAW
    severity, so run check_infirmary() on the original file, not on the fixed one.
    Other problems (see check_infirmary) need a manual decision and are not changed.
    """
    df = load_table("infirmary_visits")  # applies READ_OPTIONS (cp932) from load.py

    df["injury_code"] = df["injury_code"].str.upper().str.replace(r"^([A-Z]{3})-?(\d{2})$", r"\1-\2", regex=True)

    adm = pd.to_datetime(df["admitted_at"])
    dis = pd.to_datetime(df["discharged_at"])
    wrong_tz = adm >= UTC_DISCHARGE_FROM
    df.loc[wrong_tz, "discharged_at"] = (dis[wrong_tz] + timedelta(hours=9)).dt.strftime("%Y-%m-%d %H:%M")

    df["severity"] = 6 - df["severity"]

    return df[df["injury_code"] != "TST-00"].reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(check_infirmary(summary=True).to_string(index=False))
