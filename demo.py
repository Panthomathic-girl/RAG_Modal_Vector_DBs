import weaviate
from app.config import Settings

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=Settings.WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(Settings.WEAVIATE_API_KEY)
)

collections = client.collections.list_all()
print("Collections:", collections)

client.close()
