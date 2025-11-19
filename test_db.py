from sqlalchemy import create_engine, text
import os

db_url = os.environ.get("DATABASE_URL")
print("Using DB URL:", db_url)

engine = create_engine(db_url)

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        print("Connected OK!", result.fetchone()[0])
except Exception as e:
    print("Connection FAILED:", e)
