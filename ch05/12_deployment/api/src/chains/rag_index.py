import os
import time
from pathlib import Path

import dotenv
import requests
import tcvectordb
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.enum import ReadConsistency, FieldType, IndexType, MetricType
from tcvectordb.model.index import Index, FilterIndex, VectorIndex, SparseIndex, HNSWParams

from utils import load_documents, split_sections, split_chunks

dotenv.load_dotenv()


def get_all_splits():
    docs = load_documents("../../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


def encode_texts(texts, jina_api_key="", batch_size=32):
    url = 'https://api.jina.ai/v1/embeddings'
    jina_api_key = os.environ.get("JINA_API_KEY", jina_api_key)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {jina_api_key}",
            "Accept-Encoding": "identity",
            "Content-type": "application/json",
        }
    )
    embeddings = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        data = {
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "late_chunking": False,
            "dimensions": 1024,
            "embedding_type": "float",
            "input": batch
        }
        resp = session.post(url, json=data).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])
        cur_embeddings = resp["data"]
        sorted_embeddings = sorted(cur_embeddings, key=lambda e: e["index"])  # type: ignore
        batch_embeddings = [result["embedding"] for result in sorted_embeddings]
        embeddings.extend(batch_embeddings)
    return embeddings


DB_URL = os.environ.get("DB_URL", "")
DB_USERNAME = os.environ.get("DB_USERNAME", "root")
DB_KEY = os.environ.get("DB_KEY", "")
DB_NAME = "db-olympic-games"
COLLECTION_NAME = "olympic-games-hybrid"

client = tcvectordb.RPCVectorDBClient(
    url=DB_URL,
    key=DB_KEY,
    username=DB_USERNAME,
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30)

db_exists = False
for db in client.list_databases():
    if db.database_name == DB_NAME:
        db_exists = True
        break
if not db_exists:
    db = client.create_database(database_name=DB_NAME)
else:
    db = client.database(DB_NAME)

index = Index(
    FilterIndex(
        'id',
        FieldType.String,
        IndexType.PRIMARY_KEY),
    VectorIndex(
        'vector',
        dimension=1024,
        index_type=IndexType.HNSW,
        metric_type=MetricType.COSINE,
        params=HNSWParams(m=16, efconstruction=200)),
    SparseIndex(
        name='sparse_vector',
        field_type=FieldType.SparseVector,
        index_type=IndexType.SPARSE_INVERTED,
        metric_type=MetricType.IP),
    FilterIndex('text', FieldType.String, IndexType.FILTER),
    FilterIndex('source', FieldType.String, IndexType.FILTER),
    FilterIndex('title', FieldType.String, IndexType.FILTER),
)

try:
    coll = db.describe_collection(COLLECTION_NAME)
except tcvectordb.exceptions.VectorDBException:
    coll = db.create_collection(
        name=COLLECTION_NAME,
        shard=1,
        replicas=0,
        description='this is a collection olympic games',
        index=index
    )

chunks = get_all_splits()
bm25 = BM25Encoder.default('zh')
batch_size = 32
total_count = len(chunks)

texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
ids = [chunk.id for chunk in chunks]
sparse_embeddings = bm25.encode_texts(texts)
dense_embeddings = encode_texts(texts, batch_size=batch_size)

for start in range(0, total_count, batch_size):
    end = min(start + batch_size, total_count)
    docs = []
    for id in range(start, end, 1):
        if metadatas[id] and isinstance(metadatas[id], dict):
            metadata = metadatas[id]
        else:
            metadata = {}
        doc_id = ids[id] or "{}-{}-{}".format(time.time_ns(), hash(texts[id]), id)
        doc_attrs = {
            "id": doc_id,
            "vector": dense_embeddings[id],
            "sparse_vector": sparse_embeddings[id],
            "text": texts[id],
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
        }
        docs.append(doc_attrs)
    client.upsert(
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        documents=docs,
        timeout=30
    )

client.close()
