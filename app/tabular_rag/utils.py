from fastapi import UploadFile, HTTPException
import io
import pandas as pd
import psycopg2
from pymongo import MongoClient
from app.config import Settings

MAX_BYTES = 200 * 1024 * 1024  # 200 MB limit

def _enforce_size_limit(file: UploadFile, max_bytes: int = MAX_BYTES) -> bytes:
    """Enforce size limit on uploaded file and return content."""
    buf = io.BytesIO()
    total = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File exceeds {max_bytes//(1024*1024)} MB limit.")
        buf.write(chunk)
    file.file.seek(0)
    return buf.getvalue()

def _parse_csv(file_content: bytes) -> list:
    """Parse CSV content into a list of dictionaries with error handling."""
    try:
        df = pd.read_csv(io.BytesIO(file_content), dtype=str)  # Force string to handle mixed types
        return df.to_dict(orient='records')
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing failed: {str(e)}")

def get_postgres_connection():
    """Establish PostgreSQL connection with retry logic."""
    try:
        conn = psycopg2.connect(
            dbname=Settings.POSTGRES_DB,
            user=Settings.POSTGRES_USER,
            password=Settings.POSTGRES_PASSWORD,
            host=Settings.POSTGRES_HOST,
            port=Settings.POSTGRES_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=500, detail=f"PostgreSQL connection error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# utils.py → get_mongo_connection()
def get_mongo_connection():
    """Establish MongoDB connection with authentication."""
    try:
        client = MongoClient(
            host=Settings.MONGO_HOST,
            port=int(Settings.MONGO_PORT),
            username=Settings.MONGO_USER,
            password=Settings.MONGO_PASSWORD,
            authSource="admin"  # ← CRITICAL: admin, NOT Settings.MONGO_DB
        )
        client.server_info()  # Test connection
        return client[Settings.MONGO_DB]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {str(e)}")