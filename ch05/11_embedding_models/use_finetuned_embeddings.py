from pathlib import Path
import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

# model_name = "jinaai/jina-embeddings-v3"
model_name = "/home/jinaai/nanw/finetune_je_v3/models/jina-embeddings-v3-olympics-v1/checkpoint-1880"
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    cache_folder="/home/jinaai/nanw/finetune_je_v3/.cache",
    model_kwargs={'trust_remote_code': True}
)

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser

from utils import load_documents, format_docs, split_chunks, split_sections

import dotenv

dotenv.load_dotenv()

vector_db_dir = '../data_chroma_local_embeddings_v0'
collection_name = 'olympic_games'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=embeddings_model,
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    docs = load_documents("./data/*.txt")
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
        embedding=embeddings_model,
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
with open("qa_pairs_rewrite.json", "r") as f:
    qa_pairs = json.load(f)
    for doc in tqdm(qa_pairs):
        query = doc["query"]
        result = rag_chain.invoke(query)
        doc["response"] = {
            "content": result["answer"],
            "contexts": [result["context"],]
        }
        results.append(doc)

output_path = "ch0504_finetuned_embeddings/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/jinaai/nanw/finetune_je_v3/.cache NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python use_finetuned_embeddings.py