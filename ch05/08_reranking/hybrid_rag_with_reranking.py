import os

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from loguru import logger
from typing_extensions import List, TypedDict

dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
    create_collection_if_not_exists=False,
    collection_name=COLLECTION_NAME)

vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
ids = vector_store.get()["ids"]
logger.info(f"Retrieved {len(ids)} documents from the vector store at {VECTOR_DB_DIR}")
chunks = {
    k: v for k, v in vector_store.get(ids=ids).items() if k in ("ids", "metadatas", "documents")
}
documents = []
for t in zip(*chunks.values()):
    d = dict(zip(chunks, t))
    documents.append(
        Document(
            page_content=d["documents"],
            id=d["ids"],
            metadata=d["metadatas"]))


def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != '']
        result += split_tokens
    return result


seg = pkuseg.pkuseg()
bm25_retriever = BM25Retriever.from_documents(
    documents, preprocess_func=tokenize_doc)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
)

llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")


def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

query = "2024年巴黎奥运会的开幕式是哪一天?"
result = graph.invoke({"question": query})
