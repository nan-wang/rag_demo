import re
from pathlib import Path
import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser

from utils import load_documents, format_docs, split_chunks, split_sections

import dotenv

dotenv.load_dotenv()


vector_db_dir = '../data_chroma_local_embeddings'
collection_name = 'olympic_games'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
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
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
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

output_path = "../03_eval/data_metrics/v20241219/ch0504_local_embeddings/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
