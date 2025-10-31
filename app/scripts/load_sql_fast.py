# scripts/load_sql_fast.py
import pandas as pd
import psycopg2
from io import StringIO
from app.config import Settings   # <-- pulls DB credentials from .env

# -------------------------------------------------
# 1. Load CSV
# -------------------------------------------------
csv_path = "E:\\vs code\\chatbot\\large_test.csv"
df = pd.read_csv(csv_path)

# Ensure column order matches the table
df = df[['id', 'name', 'age', 'city']]

# -------------------------------------------------
# 2. Connect
# -------------------------------------------------
conn = psycopg2.connect(
    dbname=Settings.POSTGRES_DB,
    user=Settings.POSTGRES_USER,
    password=Settings.POSTGRES_PASSWORD,
    host=Settings.POSTGRES_HOST,
    port=Settings.POSTGRES_PORT
)
cur = conn.cursor()

# -------------------------------------------------
# 3. Optional: truncate old data
# -------------------------------------------------
cur.execute("TRUNCATE TABLE large_test RESTART IDENTITY;")
# (use DELETE if you want to keep sequence)

# -------------------------------------------------
# 4. Stream CSV → COPY (fastest bulk load)
# -------------------------------------------------
buffer = StringIO()
df.to_csv(buffer, index=False, header=False, sep=',')
buffer.seek(0)

cur.copy_expert(
    """
    COPY large_test (id, name, age, city)
    FROM STDIN WITH (FORMAT CSV, DELIMITER ',', NULL '')
    """,
    buffer
)
conn.commit()

# -------------------------------------------------
# 5. Final check
# -------------------------------------------------
cur.execute("SELECT COUNT(*) FROM large_test;")
count = cur.fetchone()[0]
print(f"Success: {count} rows inserted into PostgreSQL (large_test).")

cur.close()
conn.close()