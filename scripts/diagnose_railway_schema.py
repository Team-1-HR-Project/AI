"""Read-only Railway schema diagnostic.

This script intentionally prints no connection URL, password, API key, or token.
It only reports connectivity and schema metadata.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect, text

from app.db.session import engine

RELEVANT_TABLES = {
    "employees",
    "users",
    "departments",
    "goals",
    "evaluations",
    "evaluation_periods",
    "evaluation_scores",
    "evaluation_categories",
    "evaluation_evidence",
    "tasks",
    "task_assignments",
    "task_activities",
}


def main() -> None:
    with engine.connect() as connection:
        print("CONNECTION=success")
        print(f"DATABASE={connection.execute(text('SELECT DATABASE()')).scalar()}")
        print(f"SERVER_VERSION={connection.execute(text('SELECT VERSION()')).scalar()}")
        tables = inspect(connection).get_table_names()
        print("TABLES=" + ",".join(tables))
        for table in tables:
            if table not in RELEVANT_TABLES:
                continue
            print(f"-- {table} --")
            rows = connection.execute(
                text("""
                    SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY,
                           CHARACTER_SET_NAME, COLLATION_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table
                    ORDER BY ORDINAL_POSITION
                """),
                {"table": table},
            )
            for row in rows:
                print("|".join("" if value is None else str(value) for value in row))


if __name__ == "__main__":
    main()
