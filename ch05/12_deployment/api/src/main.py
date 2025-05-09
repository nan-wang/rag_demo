from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.rag_query import rag_graph
from models.query import QueryInput, QueryOutput
from utils.async_utils import async_retry


app = FastAPI(
    title="1980-2024年奥运会问答机器人",
    description="1980-2024年奥运会问答机器人API",
    version="0.1.0",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@async_retry(max_retries=10, delay=1)
async def invoke_qa_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await rag_graph.ainvoke(query)


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/ask")
async def query_qa(query: QueryInput) -> QueryOutput:
    query_response = await invoke_qa_with_retry(query.text)
    return query_response
