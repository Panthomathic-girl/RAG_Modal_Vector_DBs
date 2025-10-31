# app/scripts/load_mongo.py
from pymongo import MongoClient
import pandas as pd
from app.config import Settings

settings = Settings()

# Load CSV
df = pd.read_csv("large_test.csv")

# Connect to MongoDB → FIX: authSource="admin"
client = MongoClient(
    host=settings.MONGO_HOST,
    port=int(settings.MONGO_PORT),
    username=settings.MONGO_USER,
    password=settings.MONGO_PASSWORD,
    authSource="admin"  # ← THIS IS THE FIX
)

db = client[settings.MONGO_DB]
collection = db["large_test"]

# Clear & insert
collection.delete_many({})
collection.insert_many(df.to_dict("records"))

print(f"SUCCESS: {collection.count_documents({})} documents inserted")