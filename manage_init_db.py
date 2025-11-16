import sqlite3

DB_PATH = "history.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        model TEXT,
        prediction INTEGER,
        probability REAL,
        amount REAL,
        inputs TEXT
    )
""")
conn.commit()
conn.close()

print("SQLite DB initialized.")

