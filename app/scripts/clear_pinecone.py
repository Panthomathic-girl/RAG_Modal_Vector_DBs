# clear_pinecone.py
from pinecone import Pinecone
from app.config import Settings

# Initialize Pinecone
pc = Pinecone(api_key=Settings.PINECONE_API_KEY)
index = pc.Index(Settings.PINECONE_TABULAR_INDEX)

# DELETE ALL VECTORS
index.delete(delete_all=True)

print("Pinecone index cleared! All vectors removed.")