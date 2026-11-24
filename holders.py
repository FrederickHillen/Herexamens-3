"""Who is behind a sorcerer_id. Join on the holder, not on the bare ID.

The data dictionary says an ID is never reused, but it is: when a sorcerer is killed, the ID is later
given to a new sorcerer. sorcerers.csv only describes the new holder, while missions, infirmary visits
and wearables still carry the first holder's history under the same ID. So a person is (ID, period):

    first holder  "S-0615~prev"  everything up to the casualty time (dead)
    gap           None           nobody held the ID
    new holder    "S-0615"       everything from join_date (the row in sorcerers.csv)

reused_ids()              -> one row per reused ID: died_at (first holder killed), rejoined_at (new holder joined)
id_holders()              -> one row per person, with alive = True/False
assign_holder(df, ts_col) -> adds a 'holder' column to any frame with a sorcerer_id and a UTC timestamp

Run:  python holders.py
"""
from functools import lru_cache

import pandas as pd

from load import load_table

PREVIOUS = "~prev"  # holder suffix for the first (dead) holder of a reused ID


@lru_cache(maxsize=1)
def _reused_ids():
    s = load_table("sorcerers").set_index("sorcerer_id")
    m = load_table("missions").drop_duplicates()
    died = pd.to_datetime(
        m.dropna(subset=["casualties"]).assign(sorcerer_id=lambda d: d["casualties"].str.split("|"))
        .explode("sorcerer_id").groupby("sorcerer_id")["end_ts"].min())
    r = pd.DataFrame({"died_at": died, "rejoined_at": pd.to_datetime(s["join_date"]).reindex(died.index)})
    # The ID was killed in action, yet the sorcerer on file joined after that: a new person got the ID.
    return r[r["rejoined_at"] > r["died_at"]].rename_axis("sorcerer_id").reset_index()


def reused_ids():
    """One row per reused ID: sorcerer_id, died_at (first holder killed, UTC), rejoined_at (new holder joined)."""
    return _reused_ids().copy()


def id_holders():
    """One row per person: holder, sorcerer_id, name, grade, school, status, alive.
    The first holder of a reused ID is dead and not described in sorcerers.csv (name etc. unknown)."""
    s = load_table("sorcerers")
    current = s.assign(holder=s["sorcerer_id"], alive=s["status"] == "active")
    previous = reused_ids().assign(holder=lambda d: d["sorcerer_id"] + PREVIOUS, status="deceased", alive=False)
    cols = ["holder", "sorcerer_id", "name", "grade", "school", "status", "alive"]
    return pd.concat([current, previous], ignore_index=True).reindex(columns=cols)


def assign_holder(df, ts_col, id_col="sorcerer_id"):
    """Return df with a 'holder' column: the person holding id_col at ts_col (naive UTC).
    None when the record falls in the gap between two holders (nobody held the ID)."""
    out = df.copy()
    out["holder"] = out[id_col].astype(object)
    ts = pd.to_datetime(out[ts_col])
    for r in _reused_ids().itertuples():
        same = out[id_col] == r.sorcerer_id
        out.loc[same & (ts <= r.died_at), "holder"] = r.sorcerer_id + PREVIOUS
        out.loc[same & (ts > r.died_at) & (ts < r.rejoined_at), "holder"] = None
    return out


if __name__ == "__main__":
    print(reused_ids().to_string(index=False))
