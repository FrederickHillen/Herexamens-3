"""Bonus: who is the Inverter's host? Every number is computed live from the files (drop 2: audit_log).

curse_actions() -> per curse: the audit entries that caused it, the account, and the proof in the data
host()          -> the one account behind every curse, mapped to its sorcerer
host_evidence() -> the full case as text lines (also used by report.py)

Run:  python host.py
"""
from functools import lru_cache

import pandas as pd

from holders import reused_ids
from infirmary_checks import UTC_CLOCK_CHANGE
from load import DATA_DIR, load_table
from rct import FLIPPED_FIRMWARE

ROUTINE = {"field correction", "manual entry", "export for report", "routine sync"}  # 13 accounts, office hours
JST = pd.Timedelta(hours=9)


@lru_cache(maxsize=1)
def _audit():
    a = load_table("audit_log")
    if a is None:
        raise FileNotFoundError("audit_log.csv (drop 2) is not available")
    a["t"] = pd.to_datetime(a["ts"], utc=True).dt.tz_localize(None)
    a["hour_jst"] = (a["t"] + JST).dt.hour
    return a


def _v23_devices():
    """First v2.3 reading per device, straight from the wearable file."""
    w = pd.read_csv(DATA_DIR / "ce_wearable.csv", usecols=["device_id", "ts", "firmware"])
    w = w[w["firmware"] == FLIPPED_FIRMWARE]
    return pd.to_datetime(w.groupby("device_id")["ts"].min(), utc=True).dt.tz_localize(None)


def curse_actions():
    """One row per curse: matching audit entries, who made them, and whether the data confirms the link."""
    a = _audit()
    rows = []

    # 1. firmware v2.3 = the sign flip in ce_wearable
    push = a[a["detail"].str.contains(FLIPPED_FIRMWARE, regex=False)]
    first = _v23_devices()
    lag = (first.reindex(push["record_id"]).values - push["t"].values) / pd.Timedelta(hours=1)
    same = set(push["record_id"]) == set(first.index)
    rows.append(("wearable sign flip (firmware v2.3)", push,
                 same,
                 f"{push['record_id'].nunique()} devices pushed {push['t'].min():%Y-%m-%d}; exactly the "
                 f"{len(first)} devices that ever report {FLIPPED_FIRMWARE} ({'same set' if same else 'DIFFERENT set'}); "
                 f"each starts {lag.min():.0f}-{lag.max():.0f} h after its push"))

    # 2. data dictionary: severity scale description edited = the reversed scale in the dictionary
    dd = a[(a["object"] == "data_dictionary") & a["record_id"].fillna("").str.contains("severity")]
    v = load_table("infirmary_visits")
    stay = (pd.to_datetime(v["discharged_at"]) - pd.to_datetime(v["admitted_at"])).dt.total_seconds() / 3600
    med = stay[stay > 0].groupby(v["severity"]).median()
    reversed_ = med.idxmin() == 1  # in the data, 1 is the shortest stay = minor, the dictionary says critical
    rows.append(("severity scale reversed in data_dictionary.md", dd, len(dd) > 0 and reversed_,
                 f"'{dd['record_id'].iloc[0] if len(dd) else '?'}' edited; in the data severity 1 is the shortest "
                 f"stay (median {med.loc[1]:.0f} h vs {med.loc[5]:.0f} h for 5), the opposite of the dictionary"))

    # 3. infirmary_config: discharge terminal clock source = discharge stamps in UTC
    clk = a[(a["object"] == "infirmary_config") & a["detail"].str.contains("clock")]
    adm, dis = pd.to_datetime(v["admitted_at"]), pd.to_datetime(v["discharged_at"])
    neg = dis < adm
    first_neg = dis[neg].min()
    gap_min = (first_neg - clk["t"].iloc[0]).total_seconds() / 60 if len(clk) else float("nan")
    rows.append(("discharge times stamped in UTC", clk, len(clk) > 0 and abs(gap_min) < 60,
                 f"clock source changed {clk['t'].iloc[0]:%Y-%m-%d %H:%M} UTC; all {int(neg.sum())} discharges "
                 f"before admission are stamped from {first_neg:%Y-%m-%d %H:%M} on ({gap_min:+.0f} min), none earlier"))

    # 4. sorcerers: ID released and reassigned = the reused IDs (dictionary says never)
    ra = a[a["action"] == "reassign_id"]
    r = reused_ids().set_index("sorcerer_id")
    before = (r["rejoined_at"].reindex(ra["record_id"]).values - ra["t"].values) / pd.Timedelta(hours=1)
    same = set(ra["record_id"]) == set(r.index)
    rows.append(("sorcerer IDs of the dead reused", ra, same,
                 f"{len(ra)} IDs reassigned = exactly the {len(r)} reused IDs ({'same set' if same else 'DIFFERENT set'}); "
                 f"each logged {before.min():.0f}-{before.max():.0f} h before the new holder's join_date"))

    out = []
    for curse, ev, ok, proof in rows:
        out.append({"curse": curse, "entries": len(ev), "accounts": ", ".join(sorted(ev["account"].unique())),
                    "first": ev["t"].min(), "last": ev["t"].max(), "confirmed": bool(ok), "proof": proof,
                    "audit_ids": list(ev["audit_id"])})
    return pd.DataFrame(out)


def _account_to_sorcerer(account):
    """hlr.minase.i -> Medical Support sorcerer with surname Minase and first initial I (names may be reversed)."""
    prefix, surname, initial = account.split(".")
    s = load_table("sorcerers")
    def normal(name):  # undo the reversed-name curse ('esaniM irakiH' -> 'Hikari Minase')
        words = name.split()
        return name[::-1] if not words[0][0].isupper() else name
    names = s["name"].map(normal).str.split()
    hit = (names.str[-1].str.lower() == surname) & (names.str[0].str[0].str.lower() == initial)
    if prefix == "hlr":
        hit &= s["technique"] == "Medical Support"
    return s[hit]


def host():
    """The single account behind every curse action, with its sorcerer row."""
    c = curse_actions()
    accounts = set(", ".join(c["accounts"]).split(", "))
    if len(accounts) != 1 or not c["confirmed"].all():
        raise ValueError(f"no single confirmed host: accounts {accounts}, confirmed {c['confirmed'].tolist()}")
    account = accounts.pop()
    return account, _account_to_sorcerer(account)


def host_evidence():
    """The full case, as text lines."""
    a = _audit()
    c = curse_actions()
    account, s = host()
    ids = {i for lst in c["audit_ids"] for i in lst}
    mine = a[a["audit_id"].isin(ids)]
    routine = a[a["detail"].isin(ROUTINE)]
    others = a[~a["audit_id"].isin(ids) & (a["account"] != account) & ~a["detail"].isin(ROUTINE)]
    sid = ", ".join(f"{r.sorcerer_id} ({r.name}, grade {r.grade}, {r.school}, {r.technique})" for r in s.itertuples())
    lines = [f"Host: {sid}, audit account '{account}'.", ""]
    for r in c.itertuples():
        lines.append(f"- {r.curse}: {r.entries} audit entries by {r.accounts} ({r.first:%Y-%m-%d} to "
                     f"{r.last:%Y-%m-%d}). {r.proof}.")
    lines += [
        "",
        f"- Only {account} ever did any of these {len(ids)} actions; every other non-routine entry "
        f"({len(others)}: {', '.join(sorted(others['detail'].unique()))} by {', '.join(sorted(others['account'].unique()))}) "
        f"is unrelated to a curse.",
        f"- All {len(ids)} were done at night, {mine['hour_jst'].min():02d}:00-{mine['hour_jst'].max():02d}:59 JST; "
        f"all {len(routine)} routine entries of all {routine['account'].nunique()} accounts (including {account}) fall "
        f"in office hours, {routine['hour_jst'].min():02d}:00-{routine['hour_jst'].max():02d}:59 JST.",
    ]
    return lines


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print("\n".join(host_evidence()))
