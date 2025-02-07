import os

import dotenv
import tcvectordb
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank
from langchain_community.embeddings.jina import JinaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick, RunnableParallel
from langchain_openai import ChatOpenAI
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.enum import ReadConsistency

from data_models import Response
from prompts import GENERATION_PROMPT
from tcvectordb_hybrid_search import TencentVectorDBRetriever
from utils import format_docs

dotenv.load_dotenv()

DB_URL = os.environ.get("DB_URL", "")
DB_USERNAME = os.environ.get("DB_USERNAME", "root")
DB_KEY = os.environ.get("DB_KEY", "")
DB_NAME = "db-olympic-games"
COLLECTION_NAME = "olympic-games-hybrid"

bm25 = BM25Encoder.default('zh')
embeddings = JinaEmbeddings(model_name="jina-embeddings-v3")

client = tcvectordb.RPCVectorDBClient(
    url=DB_URL,
    key=DB_KEY,
    username=DB_USERNAME,
    read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
    timeout=30)

ensemble_retriever = TencentVectorDBRetriever(
    client=client,
    embeddings=embeddings,
    sparse_encoder=bm25,
    database_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    limit=10,
    weight=[0.5, 0.5],
    field_vector="vector",
    field_sparse_vector="sparse_vector",
)

compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

llm = ChatOpenAI(model="Qwen/Qwen2.5-14B-Instruct").with_structured_output(Response)
prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)

rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough())
        | RunnableParallel(
            context=RunnablePick("context"),
            question=RunnablePick("question"),
            response=prompt | llm)
)

result = rag_chain.invoke("北京奥运会是哪一年？")
print(result["response"])

client.close()
