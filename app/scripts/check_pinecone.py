# scripts/check_pinecone.py
from pinecone import Pinecone
from app.config import Settings

pc = Pinecone(api_key=Settings.PINECONE_API_KEY)
index = pc.Index(Settings.PINECONE_TABULAR_INDEX)
print(index.describe_index_stats())