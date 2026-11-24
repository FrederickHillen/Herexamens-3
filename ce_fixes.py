"""Fix ce_wearable.csv: firmware v2.3 reports ce_level with the sign flipped.

Proof (see report.py): after negating v2.3 its mean is -63.9, like v2.2 (-67.9) and v2.4 (-65.8), and
the 40 devices that ran both v2.2 and v2.3 average exactly -63.9 vs +63.9.

correct_ce(df)              -> df with ce_level corrected (a 'ce_level_raw' column keeps the original)
fix_ce_wearable(out_file)   -> writes the corrected readings to out_file (original is not touched)
"""
import numpy as np

from load import load_table

FLIPPED_FIRMWARE = "v2.3"


def correct_ce(df):
    """Return a copy of a ce_wearable DataFrame with the v2.3 sign flip undone."""
    df = df.copy()
    df["ce_level_raw"] = df["ce_level"]
    df["ce_level"] = np.where(df["firmware"] == FLIPPED_FIRMWARE, -df["ce_level"], df["ce_level"])
    return df


def fix_ce_wearable(out_file="ce_wearable_fixed.csv"):
    df = correct_ce(load_table("ce_wearable"))
    df.to_csv(out_file, index=False)
    n = int((df["firmware"] == FLIPPED_FIRMWARE).sum())
    print(f"Wrote {len(df)} rows to {out_file} ({n} v2.3 readings negated)")
    return df
