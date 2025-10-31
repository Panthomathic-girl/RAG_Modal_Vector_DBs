# app/multi_classification_RAG_modal_ZILLIZ/rag_zilliz.py
import json, logging, os
from typing import List, Dict, Any

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config (from env or config.py)
ZILLIZ_URI = os.getenv("ZILLIZ_URI")  # e.g., "https://in03-abc123.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
COLLECTION_NAME = "rag_faqs"  # Zilliz collection name
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Helper: Chunk text
def _chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Zilliz class
class ZillizRetriever:
    _client = None
    _model = None

    @classmethod
    def _load_client(cls):
        if cls._client is None:
            cls._client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_API_KEY)
        return cls._client

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_NAME)
        return cls._model

    @classmethod
    def build_index(cls, jsonl_path: str, force_rebuild: bool = False):
        client = cls._load_client()
        model = cls._load_model()

        # Drop + create collection
        if force_rebuild:
            client.drop_collection(COLLECTION_NAME)

        if client.has_collection(COLLECTION_NAME):
            logger.info("Zilliz collection exists.")
            return

        # Create collection
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=model.get_sentence_embedding_dimension())
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="IP")
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)

        # Load data
        entities = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Line {line_no} invalid – skipping")
                    continue

                text = obj.get("text", "")
                if not text:
                    continue

                chunks = _chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    vec = model.encode(chunk).tolist()
                    entities.append({
                        "vector": vec,
                        "text": chunk,
                        "title": obj.get("title", ""),
                        "section": obj.get("section", ""),
                    })

        if entities:
            client.insert(COLLECTION_NAME, entities)
            logger.info(f"Built Zilliz: {len(entities)} entities")

    @classmethod
    def retrieve(cls, query: str, top_k: int = 5) -> List[Dict]:
        client = cls._load_client()
        model = cls._load_model()
        q_vec = model.encode(query).tolist()

        res = client.search(
            COLLECTION_NAME,
            data=[q_vec],
            limit=top_k,
            output_fields=["text", "title", "section"]
        )

        results = []
        for hit in res[0]:
            results.append({
                "score": hit["distance"],
                "metadata": {
                    "text": hit["entity"]["text"],
                    "title": hit["entity"].get("title", ""),
                    "section": hit["entity"].get("section", ""),
                }
            })
        return results