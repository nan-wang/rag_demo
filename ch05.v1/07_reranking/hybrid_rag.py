import json
from pathlib import Path

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from utils import load_documents, split_sections, split_chunks, format_docs

dotenv.load_dotenv()


def get_all_splits():
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != '']
        result += split_tokens
    return result


vector_db_dir = '../data_chroma_jina_embeddings'
collection_name = 'olympic_games'
if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"{vectorstore._chroma_collection.count()} documents loaded")
else:
    chunks = get_all_splits()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=JinaEmbeddings(model_name="jina-embeddings-v3"),
        persist_directory=vector_db_dir,
        collection_name=collection_name)

vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# set up the bm25 retriever
seg = pkuseg.pkuseg()

chunks = get_all_splits()
bm25_retriever = BM25Retriever.from_documents(
    chunks, preprocess_func=tokenize_doc)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

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
        try:
            result = rag_chain.invoke(query)
            doc["response"] = {
                "content": result["answer"],
                "contexts": [result["context"], ]
            }
            results.append(doc)
        except Exception as e:
            print(f"Error: {e}")
            continue

output_path = "../03_eval/data_metrics/v20241219/ch0506_rerank/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
