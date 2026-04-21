from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


def execute_sql_file(db_path: str | Path, sql_file: str | Path) -> None:
    with sqlite3.connect(db_path) as conn:
        with open(sql_file, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


def load_csv_to_table(db_path: str | Path, csv_path: str | Path, table_name: str) -> None:
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="append", index=False)
        conn.commit()


def initialise_warehouse(db_path: str | Path, raw_csv: str | Path, feedback_csv: str | Path, ddl_sql: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    execute_sql_file(db_path, ddl_sql)
    load_csv_to_table(db_path, raw_csv, "raw_forecast_risk")
    if Path(feedback_csv).exists():
        load_csv_to_table(db_path, feedback_csv, "feedback_events")


def build_feature_mart(db_path: str | Path, mart_sql: str | Path) -> None:
    execute_sql_file(db_path, mart_sql)


def read_table(db_path: str | Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def append_feedback_event(db_path: str | Path, payload: dict) -> None:
    df = pd.DataFrame([payload])
    with sqlite3.connect(db_path) as conn:
        df.to_sql("feedback_events", conn, if_exists="append", index=False)
        conn.commit()
