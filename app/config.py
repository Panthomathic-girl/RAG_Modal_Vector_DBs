from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    # ---------- Gemini ----------
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # ---------- Pinecone ----------
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "patrika-web-index")
    PINECONE_UNSTRUCTURED_INDEX = os.getenv("PINECONE_UNSTRUCTURED_INDEX", "patrika-unstructured-index")
    PINECONE_STRUCTURED_INDEX = os.getenv("PINECONE_STRUCTURED_INDEX", "patrika-structured-index")
    PINECONE_TABULAR_INDEX = os.getenv("PINECONE_TABULAR_INDEX", "tabular-rag-index")
    PINECONE_CLASSIFICATION_INDEX = os.getenv("PINECONE_CLASSIFICATION_INDEX", "multi-classification-rag-index")
    PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

    # # ---------- Weaviate (NEW) ----------
    # WEAVIATE_URL = os.getenv("WEAVIATE_URL")  # e.g., https://your-cluster.weaviate.cloud
    # WEAVIATE_GRPC = os.getenv("WEAVIATE_GRPC")  # Optional: gRPC endpoint
    # WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    # WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "Rag_faqs")  # Default collection
    
    # ---------- Weaviate ----------  # New additions
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "MultiClassificationRAG")  # Class name for the index
    WEAVIATE_GRPC_ENDPOINT = os.getenv("WEAVIATE_GRPC_ENDPOINT") 

    # ---------- AWS ----------
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
    AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "erp-adportal-stage")

    # ---------- PostgreSQL ----------
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "6276")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "tabular_rag_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    # ---------- MongoDB ----------
    MONGO_DB = os.getenv("MONGO_DB", "tabular_rag_db")
    MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
    MONGO_PORT = os.getenv("MONGO_PORT", "27017")
    MONGO_USER = os.getenv("MONGO_USER", "rag_user")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "rag_password123")


# Misc constants
RANDOM_SEED = 42
TEST_SIZE = 0.2

BEST_MODEL_FILENAME = "app/models/best_model.pkl"
VECTORIZER_FILENAME = "app/models/intent_vectorizer.pkl"

# from dotenv import load_dotenv
# import os

# load_dotenv()

# class Settings:
#     # ---------- Gemini ----------
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#     # ---------- Pinecone ----------
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_INDEX = os.getenv("PINECONE_INDEX", "patrika-web-index")
#     PINECONE_UNSTRUCTURED_INDEX = os.getenv("PINECONE_UNSTRUCTURED_INDEX", "patrika-unstructured-index")
#     PINECONE_STRUCTURED_INDEX = os.getenv("PINECONE_STRUCTURED_INDEX", "patrika-structured-index")
#     PINECONE_TABULAR_INDEX = os.getenv("PINECONE_TABULAR_INDEX", "tabular-rag-index")
#     PINECONE_CLASSIFICATION_INDEX = os.getenv("PINECONE_CLASSIFICATION_INDEX", "multi-classification-rag-index")
#     PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
#     PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

#     # ---------- AWS ----------
#     AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
#     AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#     AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
#     AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "erp-adportal-stage")

#     # ---------- PostgreSQL ----------
#     POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
#     POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "6276")
#     POSTGRES_DB = os.getenv("POSTGRES_DB", "tabular_rag_db")
#     POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
#     POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

#     # ---------- MongoDB ----------
#     MONGO_DB = os.getenv("MONGO_DB", "tabular_rag_db")
#     MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
#     MONGO_PORT = os.getenv("MONGO_PORT", "27017")
#     MONGO_USER = os.getenv("MONGO_USER", "rag_user")
#     MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "rag_password123")

# # Misc constants
# RANDOM_SEED = 42
# TEST_SIZE = 0.2

# BEST_MODEL_FILENAME = "app/models/best_model.pkl"
# VECTORIZER_FILENAME = "app/models/intent_vectorizer.pkl"

# from dotenv import load_dotenv
# import os

# load_dotenv()

# class Settings:
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_INDEX = os.getenv("PINECONE_INDEX", "patrika-web-index")  # For chat_router
#     PINECONE_UNSTRUCTURED_INDEX = os.getenv("PINECONE_UNSTRUCTURED_INDEX", "patrika-unstructured-index")  # For unstructured_chatbot
#     PINECONE_STRUCTURED_INDEX = os.getenv("PINECONE_STRUCTURED_INDEX", "patrika-structured-index")  # For structured_multimodal_rag_chatbot
#     PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
#     PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
#     AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
#     AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
#     AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
#     AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "erp-adportal-stage")

# RANDOM_SEED = 42
# TEST_SIZE = 0.2

# BEST_MODEL_FILENAME = "app/models/best_model.pkl"
# VECTORIZER_FILENAME = "app/models/intent_vectorizer.pkl"




    
