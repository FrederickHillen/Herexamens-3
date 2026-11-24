"""Find sorcerers who may use Reverse Cursed Technique (RCT).

RCT shows up as POSITIVE cursed energy on the wearable. A visit is suspicious when the patient
healed too fast for their severity AND had positive cursed energy around the stay:

    find_rct_before()  positive CE in the window before admission
    find_rct_mission() positive CE during the mission the patient was injured in
    find_rct_during()  positive CE while admitted (admission to discharge)
    find_rct_after()   positive CE in the window after discharge

"Healed too fast": stay shorter than fast_factor x the median stay of the same severity
(fast_factor=1.0 would flag everyone below the median, which is half of all visits).

Time handling (everything is compared in UTC):
  * infirmary admitted_at          JST -> subtract 9 h
  * infirmary discharged_at        JST, but already UTC for visits admitted on/after
                                   UTC_DISCHARGE_FROM (see infirmary_checks.py) -> no shift there
  * ce_wearable.ts, device_assignments.valid_from/valid_to   already UTC
  * ce_wearable firmware v2.3 has the sign flipped, corrected by ce_fixes.correct_ce

Run:  python findRCT.py
"""
import numpy as np
import pandas as pd

from ce_fixes import correct_ce
from infirmary_checks import UTC_DISCHARGE_FROM
from load import load_table

JST = pd.Timedelta(hours=9)


def _visits():
    """Infirmary visits with correct UTC admission/discharge and stay length (hours)."""
    v = load_table("infirmary_visits")
    v = v[v["injury_code"].str.upper() != "TST-00"].copy()  # test records
    adm = pd.to_datetime(v["admitted_at"])
    dis = pd.to_datetime(v["discharged_at"])
    v["adm_utc"] = adm - JST
    v["dis_utc"] = np.where(adm >= UTC_DISCHARGE_FROM, dis, dis - JST)
    v["dis_utc"] = pd.to_datetime(v["dis_utc"])
    v["stay_h"] = (v["dis_utc"] - v["adm_utc"]).dt.total_seconds() / 3600
    v["median_h"] = v.groupby("severity")["stay_h"].transform("median")
    return v


def _readings():
    """Hourly CE readings per device (UTC), v2.3 sign corrected."""
    c = correct_ce(load_table("ce_wearable"))  # undoes the v2.3 sign flip (ce_fixes.py)
    c["ts"] = pd.to_datetime(c["ts"])
    c["ce"] = c["ce_level"]
    return {dev: (g["ts"].values, g["ce"].values) for dev, g in c.sort_values("ts").groupby("device_id")}


def _positive_in_window(visits, start_col, end_col, hours_before, hours_after):
    """For each visit: number of positive readings and max CE in [start-hours_before, end+hours_after]
    from the device(s) the patient wore at that time."""
    readings = _readings()
    a = load_table("device_assignments")
    a["valid_from"] = pd.to_datetime(a["valid_from"])
    a["valid_to"] = pd.to_datetime(a["valid_to"]).fillna(pd.Timestamp("2262-01-01"))
    by_sorcerer = {s: g for s, g in a.groupby("sorcerer_id")}

    pos, mx = [], []
    for r in visits.itertuples():
        lo = np.datetime64(getattr(r, start_col) - pd.Timedelta(hours=hours_before))
        hi = np.datetime64(getattr(r, end_col) + pd.Timedelta(hours=hours_after))
        n, best = 0, np.nan
        g = by_sorcerer.get(r.sorcerer_id)
        if g is not None:
            for dev in g[(g["valid_from"] <= getattr(r, end_col)) & (g["valid_to"] >= getattr(r, start_col))]["device_id"]:
                ts, ce = readings[dev]
                i, j = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right")
                w = ce[i:j]
                if len(w):
                    n += int((w > 0).sum())
                    best = np.nanmax([best, w.max()])
        pos.append(n)
        mx.append(best)
    return pos, mx


def _find(window_h, fast_factor, side):
    v = _visits()
    fast = v[v["stay_h"] < fast_factor * v["median_h"]].copy()
    if side == "mission":  # the mission the visit came from (missions.csv is UTC, has duplicate rows)
        m = load_table("missions").drop_duplicates("mission_id")[["mission_id", "start_ts", "end_ts"]]
        m["ms_start"], m["ms_end"] = pd.to_datetime(m["start_ts"]), pd.to_datetime(m["end_ts"])
        fast = fast.merge(m[["mission_id", "ms_start", "ms_end"]], on="mission_id", how="inner")
        fast["positive_readings"], fast["max_ce"] = _positive_in_window(fast, "ms_start", "ms_end", 0, 0)
    elif side == "before":  # window ends at admission
        fast["positive_readings"], fast["max_ce"] = _positive_in_window(fast, "adm_utc", "adm_utc", window_h, 0)
    elif side == "during":  # the whole stay
        fast["positive_readings"], fast["max_ce"] = _positive_in_window(fast, "adm_utc", "dis_utc", 0, 0)
    else:                 # window starts at discharge
        fast["positive_readings"], fast["max_ce"] = _positive_in_window(fast, "dis_utc", "dis_utc", 0, window_h)
    out = fast[fast["positive_readings"] > 0].merge(
        load_table("sorcerers")[["sorcerer_id", "name"]], on="sorcerer_id", how="left")
    return out[["visit_id", "sorcerer_id", "name", "severity", "adm_utc", "dis_utc", "stay_h", "median_h",
                "positive_readings", "max_ce"]].sort_values("positive_readings", ascending=False)


def find_rct_before(window_h=24, fast_factor=0.5):
    """Fast healers with positive cursed energy in the window_h hours BEFORE admission."""
    return _find(window_h, fast_factor, "before")


def find_rct_mission(fast_factor=0.5):
    """Fast healers with positive cursed energy during the mission they were injured in."""
    return _find(0, fast_factor, "mission")


def find_rct_during(fast_factor=0.5):
    """Fast healers with positive cursed energy between admission and discharge."""
    return _find(0, fast_factor, "during")


def find_rct_after(window_h=24, fast_factor=0.5):
    """Fast healers with positive cursed energy in the window_h hours AFTER discharge."""
    return _find(window_h, fast_factor, "after")


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 20)
    for name, f in (("BEFORE admission", find_rct_before), ("DURING the mission", find_rct_mission),
                    ("DURING the stay", find_rct_during),
                    ("AFTER discharge", find_rct_after)):
        r = f()
        print(f"\n=== Fast healers with positive CE {name}: {len(r)} visits, "
              f"{r['sorcerer_id'].nunique()} sorcerers ===")
        print(r.round({'stay_h': 1, 'median_h': 1, 'max_ce': 1}).head(15).to_string(index=False))
