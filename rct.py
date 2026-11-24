"""C2 / D1 / D2: the one definition of RCT evidence. Every other script imports it from here.

RCT_DEFINITION      -> the definition, in one sentence
is_rct_evidence(df) -> the rule itself, applied to a per-stay table (the ONLY implementation)
stay_evidence()     -> every infirmary stay with the measurements the rule needs + the verdict
rct_holders()       -> every person (ID holder, see holders.py) with enough evidence, alive or dead
rct_users()         -> D1: the living ones
lookalikes()        -> D2: sorcerers who look like RCT users but are not, with the reason
healer_effect()     -> evidence that healer sessions (drop 1) do not explain an impossibly fast recovery

All inputs are repaired first: infirmary times on one clock and severity in the dictionary's order
(infirmary_checks.clean_visits), wearable sign flip undone (corrected_ce), reused IDs split per person
(holders.py).

Run:  python rct.py
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from holders import assign_holder, id_holders
from infirmary_checks import clean_visits
from load import DATA_DIR, load_table

RCT_DEFINITION = (
    "A stay is evidence of RCT when a critical or severe injury (corrected severity 1-2) heals in less than a "
    "tenth of the median time for that severity without any healer session, while the patient's own wearable "
    "reads positive cursed energy that no healer produced (firmware v2.3 sign flip undone) for at least 90% of "
    "the stay."
)

FLIPPED_FIRMWARE = "v2.3"      # reports ce_level with the wrong sign (see report.py, B4)
MAX_SEVERITY = 2               # 1 = critical, 2 = severe (after reversing the raw scale)
MAX_RECOVERY_RATIO = 0.1       # RCT users: 0.02-0.07x the median (5-14 h). Ordinary serious stays, with or
                               # without a healer, never go below ~0.46x; the fastest lookalike with its own
                               # positive energy is 0.11x (27 h)
FAST_RATIO = 1 / 3             # D2 only: faster than any ordinary serious stay, but not RCT-fast
MIN_OWN_POSITIVE = 0.9         # positive energy for (almost) the whole stay, not explained by a healer session
MIN_EVIDENCE_STAYS = 2         # D1: a repeated pattern, not a single record
JST = pd.Timedelta(hours=9)


def corrected_ce(df):
    """ce_level with the firmware sign flip undone."""
    return np.where(df["firmware"] == FLIPPED_FIRMWARE, -df["ce_level"], df["ce_level"])


@lru_cache(maxsize=1)
def _stays():
    """Cleaned infirmary visits in UTC, with recovery time (h), ratio to the severity median and the holder."""
    v = clean_visits()
    v["adm_utc"] = pd.to_datetime(v["admitted_at"]) - JST
    v["dis_utc"] = pd.to_datetime(v["discharged_at"]) - JST
    v["recovery_h"] = (v["dis_utc"] - v["adm_utc"]).dt.total_seconds() / 3600
    v["recovery_ratio"] = v["recovery_h"] / v.groupby("severity")["recovery_h"].transform("median")
    return assign_holder(v, "adm_utc")  # the person behind the ID at admission (IDs were reused)


@lru_cache(maxsize=1)
def _worn_readings():
    """Every hourly wearable reading with the wearer (sorcerer_id, holder) and the corrected ce."""
    a = load_table("device_assignments")
    a["valid_from"], a["valid_to"] = pd.to_datetime(a["valid_from"]), pd.to_datetime(a["valid_to"])
    c = load_table("ce_wearable")
    c["ts"] = pd.to_datetime(c["ts"])
    c["ce"] = corrected_ce(c)
    c = c.merge(a, on="device_id")
    c = c[(c["ts"] >= c["valid_from"]) & (c["valid_to"].isna() | (c["ts"] < c["valid_to"]))]
    return assign_holder(c[["sorcerer_id", "device_id", "ts", "ce", "firmware"]], "ts")


def _stay_readings(v):
    """Per stay: share of the patient's readings that are positive, share that is positive AND not produced by a
    healer session for this patient (own energy), number of readings, share on firmware v2.3."""
    c = _worn_readings()
    x = v[["visit_id", "sorcerer_id", "adm_utc", "dis_utc"]].merge(c, on="sorcerer_id")
    x = x[(x["ts"] >= x["adm_utc"]) & (x["ts"] <= x["dis_utc"])]
    x = x.assign(pos=x["ce"] > 0, v23=x["firmware"] == FLIPPED_FIRMWARE)
    h = _sessions()
    if h is None:
        x["by_healer"] = False
    else:  # hours in which a healer was treating this patient: that positive energy is the healer's
        y = x.loc[x["pos"], ["visit_id", "sorcerer_id", "ts"]].merge(
            h[["patient_id", "start", "end"]], left_on="sorcerer_id", right_on="patient_id")
        y = y.loc[(y["ts"] >= y["start"].dt.floor("h")) & (y["ts"] <= y["end"]), ["visit_id", "ts"]]
        x = x.merge(y.drop_duplicates().assign(by_healer=True), on=["visit_id", "ts"], how="left")
        x["by_healer"] = x["by_healer"].fillna(False).astype(bool)
    x["own_pos"] = x["pos"] & ~x["by_healer"]
    return x.groupby("visit_id").agg(positive_share=("pos", "mean"), own_positive_share=("own_pos", "mean"),
                                     n_readings=("pos", "size"), v23_share=("v23", "mean"))


@lru_cache(maxsize=1)
def _sessions():
    """Healer sessions (drop 1) with start/end in naive UTC, or None before the drop arrives.
    Checked against the wearables: healer and patient both read positive for 100% of every session."""
    if not (DATA_DIR / "healer_sessions.csv").exists():
        return None
    h = load_table("healer_sessions")
    h["start"] = pd.to_datetime(h["start_ts"], utc=True).dt.tz_localize(None)
    h["end"] = h["start"] + pd.to_timedelta(h["duration_min"], unit="m")
    return h


def _healer_present(v):
    """True if a healer session for the patient overlaps the stay (all False before drop 1)."""
    h = _sessions()
    if h is None:
        return pd.Series(False, index=v.index)
    x = v[["visit_id", "sorcerer_id", "adm_utc", "dis_utc"]].merge(h, left_on="sorcerer_id", right_on="patient_id")
    hit = x.loc[(x["start"] < x["dis_utc"]) & (x["end"] > x["adm_utc"]), "visit_id"]
    return v["visit_id"].isin(hit)


def is_rct_evidence(stays):
    """THE rule (see RCT_DEFINITION). Input: a frame with the columns made by stay_evidence()."""
    return ((stays["severity"] <= MAX_SEVERITY)
            & (stays["recovery_ratio"] < MAX_RECOVERY_RATIO)
            & ~stays["healer_present"]
            & (stays["own_positive_share"] >= MIN_OWN_POSITIVE))


@lru_cache(maxsize=1)
def _stay_evidence():
    v = _stays()
    v = v.merge(_stay_readings(v), left_on="visit_id", right_index=True, how="left")
    cols = ["positive_share", "own_positive_share", "v23_share"]
    v[cols] = v[cols].fillna(0.0)
    v["n_readings"] = v["n_readings"].fillna(0).astype(int)
    v["healer_present"] = _healer_present(v)
    v["is_rct_evidence"] = is_rct_evidence(v)
    return v


def stay_evidence():
    """Every infirmary stay with holder, severity, recovery_ratio, positive_share, own_positive_share, v23_share,
    healer_present, is_rct_evidence."""
    return _stay_evidence().copy()


def rct_holders():
    """Every person (ID holder) with at least MIN_EVIDENCE_STAYS evidence stays, alive or dead."""
    ev = stay_evidence()
    n = ev[ev["is_rct_evidence"]].groupby("holder").size().rename("evidence_stays")
    h = id_holders().merge(n, left_on="holder", right_index=True)
    return h[h["evidence_stays"] >= MIN_EVIDENCE_STAYS].sort_values("holder").reset_index(drop=True)


def rct_users():
    """D1: living sorcerers with at least MIN_EVIDENCE_STAYS stays that count as RCT evidence."""
    h = rct_holders()
    return (h[h["alive"]][["sorcerer_id", "name", "grade", "school", "evidence_stays"]]
            .reset_index(drop=True))


def _profile():
    """Per holder: the numbers behind every D2 reason."""
    ev = stay_evidence()
    serious = ev[ev["severity"] <= MAX_SEVERITY]
    fast = serious[serious["recovery_ratio"] < FAST_RATIO]
    p = pd.DataFrame({
        "stays": ev.groupby("holder").size(),
        "full_positive_stays": ev[ev["positive_share"] >= 0.9].groupby("holder").size(),
        "serious_stays": serious.groupby("holder").size(),
        "fast_serious": fast.groupby("holder").size(),
        "fast_critical": fast[fast["severity"] <= MAX_SEVERITY].groupby("holder").size(),
        "fast_with_healer": fast[fast["healer_present"]].groupby("holder").size(),
        "fast_own_max": fast.groupby("holder")["own_positive_share"].max(),
        "evidence": ev[ev["is_rct_evidence"]].groupby("holder").size(),
        "fast_ratio_min": fast.groupby("holder")["recovery_ratio"].min(),
        "fast_ratio_max": fast.groupby("holder")["recovery_ratio"].max(),
        "fast_h_min": fast.groupby("holder")["recovery_h"].min(),
        "fast_h_max": fast.groupby("holder")["recovery_h"].max(),
        "fast_positive_max": fast.groupby("holder")["positive_share"].max(),
        "fast_v23_min": fast.groupby("holder")["v23_share"].min(),
    })
    # positive hours outside the holder's own stays (healers glow while treating others)
    pos = _worn_readings()
    pos = pos[pos["ce"] > 0]
    inside = pos.merge(ev[["holder", "adm_utc", "dis_utc"]], on="holder")
    inside = inside[(inside["ts"] >= inside["adm_utc"]) & (inside["ts"] <= inside["dis_utc"])]
    p["positive_hours"] = pos.groupby("holder").size()
    p["positive_hours_in_stays"] = inside.drop_duplicates(["holder", "ts"]).groupby("holder").size()
    counts = ["fast_serious", "fast_critical", "fast_with_healer", "evidence"]
    p = p.fillna({c: 0 for c in p.columns if c in counts or not c.startswith("fast_")})
    p["outside_share"] = 1 - p["positive_hours_in_stays"] / p["positive_hours"].replace(0, np.nan)
    h = _sessions()
    if h is not None:  # positive hours explained by sessions in which the holder was the healer
        g = pos.merge(h, left_on="sorcerer_id", right_on="healer_id")
        g = g[(g["ts"] >= g["start"].dt.floor("h")) & (g["ts"] <= g["end"])]
        p["sessions_as_healer"] = h.groupby("healer_id").size()
        p["positive_hours_healing"] = g.drop_duplicates(["holder", "ts"]).groupby("holder").size()
        p = p.fillna({"sessions_as_healer": 0, "positive_hours_healing": 0})
    return p


def lookalikes():
    """D2: every person who looks like an RCT user but is not, with the reason in data."""
    users = set(rct_users()["sorcerer_id"])
    p = _profile().join(id_holders().set_index("holder")[["sorcerer_id", "name", "alive"]])
    dead_rct = set(rct_holders().query("~alive")["holder"])
    rows = []
    for holder, r in p.iterrows():
        if holder in users:
            continue
        if holder in dead_rct:
            reason = (f"real RCT evidence, but this is the first holder of {r['sorcerer_id']}, killed in action; "
                      f"the ID was given to a new sorcerer (see holders.py)")
        elif r["fast_serious"] > 0 and r["fast_own_max"] == 0 and r["fast_with_healer"] == r["fast_serious"]:
            reason = (f"{int(r['fast_serious'])} serious injuries healed in {r['fast_h_min']:.0f}-{r['fast_h_max']:.0f} h, "
                      f"but a healer treated every one of them and all positive energy in those stays came from "
                      f"the healer sessions (none of it is their own)")
        elif r["fast_serious"] > 0 and r["fast_positive_max"] == 0:
            reason = (f"{int(r['fast_serious'])} serious injuries healed in {r['fast_h_min']:.0f}-{r['fast_h_max']:.0f} h, "
                      f"but the own wearable is never positive during them"
                      + (f" (all on firmware {FLIPPED_FIRMWARE}: raw readings only look positive because of the "
                         f"sign flip)" if r["fast_v23_min"] > 0.5 else ""))
        elif r["fast_serious"] > 0 and r["fast_ratio_min"] >= MAX_RECOVERY_RATIO:
            reason = (f"own positive energy, but serious injuries heal in {r['fast_h_min']:.0f}-{r['fast_h_max']:.0f} h "
                      f"({r['fast_ratio_min']:.2f}-{r['fast_ratio_max']:.2f}x the median): faster than normal, not the "
                      f"<{MAX_RECOVERY_RATIO}x (hours instead of days) of the RCT users")
        elif 0 < r["evidence"] < MIN_EVIDENCE_STAYS:
            reason = (f"only {int(r['evidence'])} evidence stay; at least {MIN_EVIDENCE_STAYS} are required")
        elif r["fast_serious"] > 0:
            reason = (f"fast serious recoveries ({r['fast_h_min']:.0f}-{r['fast_h_max']:.0f} h) but not the full "
                      f"signature: own positive energy at most {r['fast_own_max']:.0%} of the stay, "
                      f"healer present in {int(r['fast_with_healer'])} of them")
        elif r["stays"] > 0 and r["full_positive_stays"] == r["stays"]:
            reason = (f"positive CE during all {int(r['stays'])} stays, but never a critical or severe injury, "
                      f"so no recovery to prove it")
        elif r.get("sessions_as_healer", 0) > 0:
            reason = (f"healer: {int(r['sessions_as_healer'])} sessions treating others; "
                      f"{r['positive_hours_healing'] / r['positive_hours']:.0%} of the {int(r['positive_hours'])} "
                      f"positive hours fall in those sessions, no impossibly fast recovery")
        elif r["positive_hours"] >= 100 and r["outside_share"] > 0.5:
            reason = (f"healer: {int(r['positive_hours'])} positive hours, {r['outside_share']:.0%} outside own "
                      f"stays (bursts while treating patients), no impossibly fast recovery")
        else:
            continue
        rows.append((holder, r["sorcerer_id"], r["name"], reason))
    return pd.DataFrame(rows, columns=["holder", "sorcerer_id", "name", "why_not_rct"])


def healer_effect():
    """Serious stays with and without a healer session (drop 1).
    Returns (table per severity, holders with a fast serious recovery during which a healer treated them)."""
    ev = stay_evidence()
    serious = ev[ev["severity"] <= MAX_SEVERITY]
    t = (serious.groupby(["severity", "healer_present"])["recovery_ratio"]
         .agg(stays="size", median="median", fastest="min").round(3).reset_index())
    treated_fast = serious[serious["healer_present"] & (serious["recovery_ratio"] < MAX_RECOVERY_RATIO)]
    return t, sorted(treated_fast["holder"].unique())


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print("RCT evidence:", RCT_DEFINITION, "\n")
    ev = stay_evidence()
    print(f"{int(ev['is_rct_evidence'].sum())} of {len(ev)} stays count as RCT evidence.\n")
    print("D1 - living RCT users:")
    print(rct_users().to_string(index=False), "\n")
    print("D2 - lookalikes:")
    print(lookalikes().to_string(index=False), "\n")
    t, treated_fast = healer_effect()
    print("Healer sessions vs recovery (serious stays):")
    print(t.to_string(index=False))
    print("Fast serious recoveries with a healer present:", ", ".join(treated_fast) or "none")
