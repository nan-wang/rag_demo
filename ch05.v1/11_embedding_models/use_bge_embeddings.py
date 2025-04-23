import os
from pathlib import Path
import json
from tqdm import tqdm

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser

from utils import load_documents, format_docs, split_chunks, split_sections

import dotenv

dotenv.load_dotenv()

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

vector_db_dir = '../data_chroma_bge_embeddings'
collection_name = 'olympic_games'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=OpenAIEmbeddings(
            model="Pro/BAAI/bge-m3",
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
            chunk_size=16,
            check_embedding_ctx_length=False
        ),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(
            model="Pro/BAAI/bge-m3",
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
            chunk_size=16,
            check_embedding_ctx_length=False
        ),
        persist_directory=vector_db_dir,
        collection_name=collection_name)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
)

# keep the context and return
rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough())
        | RunnableParallel(
    context=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

results = []
with open("../03_eval/data_eval/v20241219/qa_pairs_rewrite.json", "r") as f:
    qa_pairs = json.load(f)
    for doc in tqdm(qa_pairs):
        query = doc["query"]
        result = rag_chain.invoke(query)
        doc["response"] = {
            "content": result["answer"],
            "contexts": [result["context"],]
        }
        results.append(doc)

output_path = "../03_eval/data_metrics/v20241219/ch0504_bge_embeddings/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
