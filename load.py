"""Load all CSV files into an in-memory SQLite database so they can be queried with SQL.

Usage in a script / notebook:
    from load import get_connection, query
    df = query("SELECT ward, COUNT(*) AS n FROM missions GROUP BY ward")

Interactive shell:
    python load.py            # then type SQL, ';' optional, 'tables' / 'quit' also work
One-off:
    python load.py "SELECT * FROM sorcerers LIMIT 5"
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

# Per-file read options. The infirmary export uses the Japanese locale (cp932 / Shift-JIS).
READ_OPTIONS = {
    "infirmary_visits": {"encoding": "cp932"},
}

# Columns parsed as timestamps (stored in SQLite as ISO text).
# Note: infirmary timestamps are JST (UTC+9); everything else is UTC.
TIMESTAMP_COLUMNS = {
    "missions": ["start_ts", "end_ts"],
    "device_assignments": ["valid_from", "valid_to"],
    "ce_wearable": ["ts"],
    "barrier_sensors": ["ts"],
}


def load_table(name):
    """Read one CSV (by table name, without extension) into a DataFrame."""
    path = DATA_DIR / f"{name}.csv"
    try:
        df = pd.read_csv(path, **READ_OPTIONS.get(name, {}))
    except FileNotFoundError:
        print(f"Error: {path} was not found.", file=sys.stderr)
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: {path} is empty.", file=sys.stderr)
        return None
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        print(f"Error: could not parse {path}: {e}", file=sys.stderr)
        return None
    for col in TIMESTAMP_COLUMNS.get(name, []):
        # Normalise to 'YYYY-MM-DD HH:MM:SS' so SQLite date functions work.
        df[col] = pd.to_datetime(df[col], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def get_connection(data_dir=DATA_DIR):
    """Return a SQLite connection with every *.csv in data_dir loaded as a table."""
    con = sqlite3.connect(":memory:")
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = load_table(path.stem)
        if df is not None:
            df.to_sql(path.stem, con, index=False)
    return con


def query(sql, con=None, params=None):
    """Run a SQL query and return the result as a DataFrame."""
    con = con or _default_connection()
    return pd.read_sql_query(sql, con, params=params)


_con = None


def _default_connection():
    global _con
    if _con is None:
        _con = get_connection()
    return _con


def _tables(con):
    return pd.read_sql_query(
        "SELECT m.name AS table_name, "
        "(SELECT COUNT(*) FROM pragma_table_info(m.name)) AS columns "
        "FROM sqlite_master m WHERE m.type = 'table' ORDER BY 1",
        con,
    )


def repl():
    con = _default_connection()
    pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 60)
    print("Tables loaded:\n", _tables(con).to_string(index=False))
    print("\nType SQL (one line), 'tables', 'schema <table>' or 'quit'.")
    while True:
        try:
            line = input("sql> ").strip().rstrip(";")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        try:
            if line.lower() == "tables":
                print(_tables(con).to_string(index=False))
            elif line.lower().startswith("schema "):
                print(query(f"PRAGMA table_info({line.split(None, 1)[1]})", con).to_string(index=False))
            else:
                print(query(line, con).to_string(index=False))
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pd.set_option("display.width", 200, "display.max_columns", 50)
        print(query(" ".join(sys.argv[1:])).to_string(index=False))
    else:
        repl()
