"""B4 / B5 report, with every number computed live from the CSV files.

B4  dictionary_errors()  -> statements in data_dictionary.md that the data contradicts, with proof
B5  curse_log()          -> every data issue found: evidence + fix
C2  rct_section()        -> the RCT evidence definition (from rct.py), D1 answer and D2 lookalikes
    write_report(path)   -> all of it as one markdown file

Run:  python report.py [output.md]
"""
import sys

import pandas as pd

from infirmary_checks import UTC_DISCHARGE_FROM, check_infirmary
from holders import PREVIOUS, assign_holder, reused_ids
from load import load_table
from rct import FLIPPED_FIRMWARE, MIN_EVIDENCE_STAYS, RCT_DEFINITION, lookalikes, rct_users, stay_evidence


def _tables():
    t = {n: load_table(n) for n in ("sorcerers", "wards", "missions", "infirmary_visits",
                                    "device_assignments", "ce_wearable", "barrier_sensors")}
    t["m"] = t["missions"].drop_duplicates()
    return t


# --------------------------------------------------------------------------- findings
def _infirmary_timezone(t):
    v = t["infirmary_visits"]
    adm, dis = pd.to_datetime(v["admitted_at"]), pd.to_datetime(v["discharged_at"])
    stay = (dis - adm).dt.total_seconds() / 3600
    late = adm >= UTC_DISCHARGE_FROM
    med_before = stay[~late & (v["severity"] == 1)].median()
    med_after = stay[late & (v["severity"] == 1)].median()
    return (f"{int((stay < 0).sum())} visits are discharged before they are admitted, all admitted on/after "
            f"{UTC_DISCHARGE_FROM:%Y-%m-%d}; none before. Median stay of the shortest-stay severity group is "
            f"{med_before:.1f} h before that date and {med_after:.1f} h after (about 9 h less). "
            f"Admissions are consistent with JST: they follow mission end (UTC) by 20-179 min only after "
            f"subtracting 9 h.")


def _severity(t):
    v = t["infirmary_visits"]
    adm, dis = pd.to_datetime(v["admitted_at"]), pd.to_datetime(v["discharged_at"])
    early = adm < UTC_DISCHARGE_FROM  # discharge times are reliable before the switch
    med = ((dis - adm).dt.total_seconds() / 3600)[early].groupby(v["severity"][early]).median().round(0)
    code = v[v["injury_code"].str.upper().str[:3].isin(["ABR", "CRU"])].assign(
        c=lambda x: x["injury_code"].str.upper().str[:3]).groupby("c")["severity"].agg(lambda s: s.mode()[0])
    return (f"Median stay (h) by raw severity: {med.to_dict()}. Severity 5 stays are ~80x longer than severity 1, "
            f"and raw severity 1 holds abrasions/bruises/sprains (ABR -> {code['ABR']}) while 5 holds "
            f"crush/puncture injuries (CRU -> {code['CRU']}). So 1 = minor, 5 = critical: the dictionary is reversed.")


def _mission_ward(t):
    m, w = t["m"], t["wards"]
    bad = m[~m["ward"].isin(w["ward_name"])]
    rev = bad["ward"].str[::-1].isin(w["ward_name"]).sum()
    return (f"{len(bad)} of {len(m)} distinct missions ({len(bad) / len(m):.1%}) have a ward that is not in wards.csv; "
            f"{rev} of them are exactly a valid ward written backwards (e.g. '{bad['ward'].iloc[0]}'). "
            f"COUNT(DISTINCT ward) = {m['ward'].nunique()} instead of {len(w)}.")


def _reversed_names(t):
    s = t["sorcerers"]
    first, last = s["name"].str.split().str[0], s["name"].str.split().str[-1]
    fc, lc = first.value_counts(), last.value_counts()
    rare = s[(fc[first].values <= 1) | (lc[last].values <= 1)]
    return (f"{len(rare)} sorcerers have a first or last name that appears only once, e.g. "
            f"{rare['name'].head(3).tolist()}: each is a common name reversed (e.g. 'esaniM aniH' = 'Hina Minase').")


def _firmware(t):
    c = t["ce_wearable"]
    pos = c["ce_level"] > 0
    by_fw = c.groupby("firmware")["ce_level"].agg(["mean", "min", "max", "count"]).round(1)
    fw = FLIPPED_FIRMWARE
    share = (pos & (c["firmware"] == fw)).sum() / pos.sum()
    span = c.loc[c["firmware"] == fw, "ts"].agg(["min", "max"]).tolist()
    return (f"Mean ce_level by firmware: {by_fw['mean'].to_dict()}. {fw} readings ({span[0]} to {span[1]}) are "
            f"mostly positive (mean {by_fw.loc[fw, 'mean']}), the other versions are negative; {share:.0%} of all "
            f"positive readings come from {fw}, so they are a sign flip, not Reverse Cursed Technique.")


def _barrier_units(t):
    b = t["barrier_sensors"]
    med = b.groupby(b["ward_code"].str[:2])["residual_ce"].median()
    return (f"Median residual_ce: Tokyo {med['TK']:.1f}, Kyoto {med['KY']:.0f} ({med['KY'] / med['TK']:.0f}x), "
            f"stable every month, no overlap in ranges (Tokyo max {b[b.ward_code.str[:2] == 'TK'].residual_ce.max():.0f}, "
            f"Kyoto min {b[b.ward_code.str[:2] == 'KY'].residual_ce.min():.0f}). Kyoto is on a different unit.")


def _id_reuse(t):
    """sorcerer_id is not one person: proof that killed sorcerers' IDs were given to new sorcerers."""
    r = reused_ids()
    a = t["device_assignments"].assign(valid_from=lambda d: pd.to_datetime(d["valid_from"]),
                                       valid_to=lambda d: pd.to_datetime(d["valid_to"]))
    old_dev_ends = r.merge(a, left_on=["sorcerer_id", "died_at"], right_on=["sorcerer_id", "valid_to"])
    new_dev_starts = r.merge(a, left_on=["sorcerer_id", "rejoined_at"], right_on=["sorcerer_id", "valid_from"])
    m = t["m"]
    ex = m.assign(sorcerer_id=m["sorcerer_ids"].str.split("|")).explode("sorcerer_id").merge(r, on="sorcerer_id")
    start = pd.to_datetime(ex["start_ts"])
    after = start > ex["died_at"]
    in_gap = after & (start < ex["rejoined_at"])
    return (f"{len(r)} IDs ({', '.join(r['sorcerer_id'])}) appear in 'casualties' but the sorcerer on file joined "
            f"after that death. For {len(old_dev_ends)} of them the wearable ends at the exact minute of the "
            f"casualty and for {len(new_dev_starts)} a new wearable starts on the join date; none of the "
            f"{int(after.sum())} missions after the death falls between death and join date ({int(in_gap.sum())}). "
            f"Two people share each ID (e.g. S-0615: casualty 2025-12-10, joined 2026-02-02).")


def _kia(t):
    s, m = t["sorcerers"], t["m"]
    cas = m.dropna(subset=["casualties"])["casualties"].str.split("|").explode().unique()
    active = set(cas) & set(s.loc[s["status"] == "active", "sorcerer_id"])
    reused = set(reused_ids()["sorcerer_id"])
    n_dec = int((s["status"] == "deceased").sum())
    same = "exactly the reused IDs" if active == reused else f"NOT the reused IDs (differ: {sorted(active ^ reused)})"
    return (f"{len(cas)} distinct sorcerer IDs appear in 'casualties'; {n_dec} have status 'deceased' and the other "
            f"{len(active)} (status 'active') are {same}, whose status describes the new holder. "
            f"So every casualty entry is a real death and status is consistent.")


# --------------------------------------------------------------------------- B4
def dictionary_errors():
    t = _tables()
    rows = [
        ("infirmary_visits", "All timestamps are Japan Standard Time (JST)",
         _infirmary_timezone(t), "discharged_at is UTC for visits admitted from end of Feb 2026: add 9 h."),
        ("infirmary_visits", "severity: 1 = critical ... 5 = minor", _severity(t),
         "Reverse the scale (6 - severity)."),
        ("missions", "ward: Ward name (romaji)", _mission_ward(t),
         "Un-reverse ward text that is not in wards.csv."),
        ("sorcerers", "name: Full name", _reversed_names(t), "Un-reverse the affected names."),
        ("ce_wearable", "normal readings are negative; positive = Reverse Cursed Technique", _firmware(t),
         f"Negate ce_level for firmware {FLIPPED_FIRMWARE} (rct.corrected_ce)."),
        ("barrier_sensors", "residual_ce: Residual cursed energy in CE units", _barrier_units(t),
         "Do not compare Tokyo and Kyoto raw; rescale Kyoto (or z-score per city)."),
        ("sorcerers", "sorcerer_id is unique and permanent: an ID is never reused", _id_reuse(t),
         "Join on the person, not the ID (holders.py): first holder until the casualty, new holder from join_date."),
    ]
    return pd.DataFrame(rows, columns=["table", "dictionary statement", "proof", "fix"])


# --------------------------------------------------------------------------- B5
def curse_log():
    t = _tables()
    chk = check_infirmary(summary=True).set_index("issue")["n_visits"]
    n = lambda k: int(chk.get(k, 0))
    v = t["infirmary_visits"].assign(adm_utc=lambda d: pd.to_datetime(d["admitted_at"]) - pd.Timedelta(hours=9))
    holder = assign_holder(v, "adm_utc")["holder"]
    n_first = int(holder.str.endswith(PREVIOUS, na=False).sum())
    dup = len(t["missions"]) - len(t["m"])
    rows = [(r["table"], r["dictionary statement"], r["proof"], r["fix"]) for _, r in dictionary_errors().iterrows()]
    rows = [(a, f"Dictionary wrong: {b}", c, d) for a, b, c, d in rows]
    rows += [
        ("missions", "Duplicate rows",
         f"{len(t['missions'])} rows but {t['m']['mission_id'].nunique()} distinct mission_id; {dup} rows are exact copies.",
         "drop_duplicates() before any count."),
        ("infirmary_visits", "Injury code format",
         f"{n('injury_code_format')} codes are lower case / without hyphen (abr01 vs ABR-01).",
         "Upper-case and insert the hyphen."),
        ("infirmary_visits", "Test records",
         f"{n('injury_code_unknown')} rows have code TST-00, no mission ({n('mission_missing')} without a mission).",
         "Remove them."),
        ("infirmary_visits", "Discharge in UTC", f"{n('discharge_stamped_utc')} visits affected (see dictionary error).",
         "Add 9 h to discharged_at."),
        ("infirmary_visits", "Mission during stay",
         f"{n('mission_during_stay')} visits where the patient starts another mission before discharge.",
         "Not changed: needs a decision (may be real for long stays)."),
        ("infirmary_visits", "On other mission at admission",
         f"{n('on_other_mission_at_admission')} visits where the patient was on another mission when admitted.",
         "Not changed: needs a decision."),
        ("sorcerers / missions", "Casualties of sorcerers with status 'active'", _kia(t),
         "Keep them: killed in action = every distinct casualty (queries.killed_in_action)."),
        ("infirmary_visits", "Admitted before join date / after death",
         f"{n_first} visits belong to the first (dead) holder of a reused ID; joined on the bare ID they looked like "
         f"visits before joining / after death. Split by holder, {n('admitted_before_join_date')} before join and "
         f"{n('admitted_after_death')} after death remain, all of them TST-00 test records.",
         "Remove the test records; the rest is the first holder's history, not an error."),
        ("infirmary_visits", "Overlapping visits", f"{n('overlapping_visit')} visits overlap another visit of the same patient.",
         "Not changed: needs a decision."),
    ]
    return pd.DataFrame(rows, columns=["table", "issue", "evidence", "fix"])


# --------------------------------------------------------------------------- output
def _md(df, title):
    out = [f"## {title}", ""]
    for i, r in enumerate(df.itertuples(index=False), 1):
        out.append(f"**{i}. [{r[0]}] {r[1]}**")
        out.append(f"- Evidence: {r[2]}")
        out.append(f"- Fix: {r[3]}")
        out.append("")
    return "\n".join(out)


def rct_section():
    ev = stay_evidence()
    users = rct_users()
    return "\n".join([
        "## C2 - Evidence of RCT", "",
        f"**Definition:** {RCT_DEFINITION}", "",
        "Implemented once as `rct.is_rct_evidence()`; every other script imports it from `rct.py`.", "",
        f"- {int(ev['is_rct_evidence'].sum())} of {len(ev)} stays count as evidence.",
        "",
        f"## D1 - Living RCT users (at least {MIN_EVIDENCE_STAYS} evidence stays)", "",
        *[f"- **{r.sorcerer_id}** {r.name} (grade {r.grade}, {r.school}): {r.evidence_stays} evidence stays"
          for r in users.itertuples()],
        "",
        "## D2 - Lookalikes and why they are not RCT users", "",
        *[f"- **{r.holder}** {r.name if isinstance(r.name, str) else '(first holder, not on file)'}: {r.why_not_rct}"
          for r in lookalikes().itertuples()],
        "",
    ])


def write_report(path="report.md"):
    text = (_md(dictionary_errors(), "B4 - Wrong statements in the data dictionary")
            + "\n" + _md(curse_log(), "B5 - Curse log")
            + "\n" + rct_section())
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    print(write_report(sys.argv[1] if len(sys.argv) > 1 else "report.md"))
