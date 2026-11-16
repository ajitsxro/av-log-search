"""
db_check.py

Connects to the local Postgres container and prints its version.
Confirms Python can reach the database.
"""

import psycopg

DATABASE_URL = "postgresql://av:av@localhost:5432/avsearch"


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute("SELECT version()").fetchone()
        print(row[0])


if __name__ == "__main__":
    main()