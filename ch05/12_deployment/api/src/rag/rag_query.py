import os
from typing import List
from typing import TypedDict

import dotenv
import tcvectordb
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from pydantic import BaseModel, Field
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.enum import ReadConsistency

from prompts import GENERATION_PROMPT
from tcvectordb_hybrid_search import TencentVectorDBRetriever
from utils import format_docs

dotenv.load_dotenv()

DB_URL = os.environ.get("DB_URL", "")
DB_USERNAME = os.environ.get("DB_USERNAME", "root")
DB_KEY = os.environ.get("DB_KEY", "")
DB_NAME = "db-olympic-games"
COLLECTION_NAME = "olympic-games-hybrid"


class Response(BaseModel):
    selected_content: str = Field(...,
                                  description="selected content from the context that is useful to answer the question.")
    answer: str = Field(..., description="the final answer to the question.")


class State(TypedDict):
    question: str
    answer: str
    context: List[Document]


client = tcvectordb.RPCVectorDBClient(
    url=DB_URL,
    key=DB_KEY,
    username=DB_USERNAME,
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30)

ensemble_retriever = TencentVectorDBRetriever(
    client=client,
    embeddings=JinaEmbeddings(model_name="jina-embeddings-v3"),
    sparse_encoder=BM25Encoder.default('zh'),
    database_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    limit=10,
    weight=[0.5, 0.5],
    field_vector="vector",
    field_sparse_vector="sparse_vector",
)

retriever = ContextualCompressionRetriever(
    base_compressor=JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10),
    base_retriever=ensemble_retriever
)

llm = ChatOpenAI(model="Qwen/Qwen2.5-14B-Instruct").with_structured_output(Response)
prompt_template = ChatPromptTemplate.from_template(GENERATION_PROMPT)


def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    prompt = prompt_template.invoke(
        {
            "question": state["question"],
            "context": format_docs(state["context"])
        }
    )
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
rag_graph = graph_builder.compile()

client.close()
