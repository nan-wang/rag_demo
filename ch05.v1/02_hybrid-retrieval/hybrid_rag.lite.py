from pathlib import Path

import dotenv
import pkuseg
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever

from utils import load_documents, split_sections, split_chunks

dotenv.load_dotenv()


def get_chunks():
    docs = load_documents("../data/*.txt")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


vector_db_dir = "../data_chroma_jina_embeddings"
collection_name = "olympic_games"
chunks = get_chunks()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=JinaEmbeddings(model_name="jina-embeddings-v3"),
    persist_directory=VECTOR_DB_DIR,
    collection_name=COLLECTION_NAME,
)

vector_retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)


# set up the bm25 retriever
def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != ""]
        result += split_tokens
    return result


seg = pkuseg.pkuseg()
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize_doc)
bm25_retriever.k = 5

retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
)
