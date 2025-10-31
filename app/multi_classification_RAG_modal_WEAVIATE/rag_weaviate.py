# app/multi_classification_RAG_modal_WEAVIATE/rag_weaviate.py

import time
import re
import logging
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, Configure, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
from app.config import Settings
from app.multi_classification_RAG_modal_WEAVIATE.embeddings import embed_text, embed_texts
from weaviate.classes.data import DataObject
from typing import List, Dict

# --- GLOBAL VARIABLES ---
_client: WeaviateClient | None = None
_collection = None
COLLECTION_NAME = "MultiClassificationRAG"

def get_vector_store():
    global _client, _collection
    settings = Settings()

    if _client is None:
        try:
            # Extract hosts
            http_host = settings.WEAVIATE_URL.replace("https://", "").split("/")[0]
            grpc_full = settings.WEAVIATE_GRPC_ENDPOINT
            grpc_host = grpc_full.split(":")[0]
            grpc_port = int(grpc_full.split(":")[1]) if ":" in grpc_full else 443

            _client = WeaviateClient(
                connection_params=ConnectionParams(
                    http=ProtocolParams(host=http_host, port=443, secure=True),
                    grpc=ProtocolParams(host=grpc_host, port=grpc_port, secure=True)
                ),
                auth_client_secret=AuthApiKey(settings.WEAVIATE_API_KEY),
                additional_headers={"X-Google-Studio-Remainder": "true"},
                additional_config=None  # gRPC works — no skip needed
            )
            _client.connect()
            logging.info(f"Connected to Weaviate Cloud with gRPC: {grpc_host}:{grpc_port}")
        except Exception as e:
            logging.error(f"Weaviate connection failed: {e}")
            raise RuntimeError("Weaviate connection failed") from e

    if _collection is None:
        collections = _client.collections
        if not collections.exists(COLLECTION_NAME):
            logging.info(f"Creating collection: {COLLECTION_NAME}")
            collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(),
                properties=[
                    Property(name="filename",      data_type=DataType.TEXT),
                    Property(name="source",        data_type=DataType.TEXT),
                    Property(name="text",          data_type=DataType.TEXT),
                    Property(name="created_at",    data_type=DataType.INT),
                    Property(name="chunk_index",   data_type=DataType.INT),
                    Property(name="total_chunks",  data_type=DataType.INT),
                    Property(name="category",      data_type=DataType.TEXT),
                    Property(name="section",       data_type=DataType.TEXT),
                    Property(name="source_id",     data_type=DataType.TEXT),
                    Property(name="language",      data_type=DataType.TEXT),
                ],
            )
        _collection = collections.get(COLLECTION_NAME)
    return _client, _collection


def _chunk_text(text: str, chunk_size: int = 8000) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len((current + para).encode("utf-8")) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = para
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for s in sentences:
                    if len((current + s).encode("utf-8")) > chunk_size:
                        if current:
                            chunks.append(current.strip())
                            current = s
                        else:
                            chunks.append(s[:chunk_size])
                    else:
                        current += (" " + s) if current else s
        else:
            current += ("\n\n" + para) if current else para

    if current:
        chunks.append(current.strip())

    logging.info(f"Chunked into {len(chunks)} parts")
    return chunks


def upsert_texts(
    text: str,
    filename: str,
    file_type: str,
    source: str = "text",
    extra_metadata: dict | None = None,
) -> bool:
    try:
        client, collection = get_vector_store()
        text_bytes = len(text.encode("utf-8"))
        logging.info(f"Processing {text_bytes} bytes from {filename}")

        chunks = _chunk_text(text, chunk_size=8000) if text_bytes > 10_000 else [text]
        embeddings = embed_texts(chunks, task="retrieval_document")

        failed_count = 0
        successful_count = 0

        # gRPC-safe: Use batch context + add_object one-by-one
        with collection.batch.fixed_size(batch_size=50) as batch:
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
                if not vec or len(vec) != 768:
                    logging.warning(f"Skipping chunk {i}: invalid embedding")
                    failed_count += 1
                    continue

                uuid_seed = f"{extra_metadata.get('title','')}|{extra_metadata.get('section','')}|{i}"
                uid = generate_uuid5(uuid_seed)

                meta = {
                    "filename": filename,
                    "file_type": file_type,
                    "text": chunk[:500] + ("..." if len(chunk) > 500 else ""),
                    "source": source,
                    "created_at": int(time.time()),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "category": extra_metadata.get("title", "") if extra_metadata else "",
                    "section": extra_metadata.get("section", "") if extra_metadata else "",
                    "source_id": extra_metadata.get("id", "") if extra_metadata else "",
                    "language": "hi" if any(ord(c) > 127 for c in chunk) else "en",
                }

                try:
                    batch.add_object(
                        properties=meta,
                        vector=vec,
                        uuid=uid
                    )
                    successful_count += 1
                except Exception as e:
                    logging.error(f"Failed to insert chunk {i}: {e}")
                    failed_count += 1

        logging.info(f"Inserted {successful_count} objects via gRPC (failed: {failed_count})")
        return failed_count == 0

    except Exception as e:
        logging.error(f"Upsert error: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return False


def query_vector_store(text: str, top_k: int = 5, filter: Dict | None = None) -> Dict:
    try:
        client, collection = get_vector_store()
        vec = embed_text(text, task="retrieval_query")
        if not vec or len(vec) != 768:
            logging.warning("Query embedding failed")
            return {"matches": []}

        where_filter = None
        if filter:
            operands = []
            for k, v in filter.items():
                val = v["$eq"] if isinstance(v, dict) and "$eq" in v else v
                operands.append(Filter.by_property(k).equal(str(val)))
            where_filter = Filter.and_(*operands) if len(operands) > 1 else operands[0] if operands else None

        resp = collection.query.near_vector(
            near_vector=vec,
            limit=top_k,
            filters=where_filter,
            return_metadata=MetadataQuery(distance=True)
        )

        matches = []
        for obj in resp.objects:
            matches.append({
                "id": str(obj.uuid),
                "score": round(obj.metadata.distance, 4),
                "metadata": obj.properties,
            })
        logging.info(f"Query returned {len(matches)} matches via gRPC")
        return {"matches": matches}

    except Exception as e:
        logging.error(f"Query error: {e}")
        return {"matches": []}