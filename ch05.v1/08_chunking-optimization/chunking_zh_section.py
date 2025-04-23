import re
from pathlib import Path
import json
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser

from utils import load_documents

import dotenv

dotenv.load_dotenv()


def format_docs(docs):
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx+1}]{doc_str}")
    return "\n\n".join(output_list)

def split_sections(doc: Document):
    text = doc.page_content
    sections = []
    pattern = r'(==+)(.*?)==+\s*([^=]*)'

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    matches = re.finditer(pattern, text, re.DOTALL)

    cur_metadata = doc.metadata.copy()
    for match in matches:
        level = len(match.group(1)) - 1  # Determine the section level by the number of '='
        section_title = match.group(2).strip()
        content = match.group(3).strip()

        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
        section_counters[level] += 1

        cur_metadata["title"] = section_title
        sections.append(Document(page_content=content, metadata=cur_metadata))
    return sections


def split_chunks(docs):
    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=['。', '！', '？', '\?', '\n\n', '\n', '\n\n\n'],
        is_separator_regex=True,
        keep_separator="end"
    )
    return text_splitter.split_documents(docs)


vector_db_dir = '../data_chroma_zh_section'
collection_name = 'olympic_games'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=OpenAIEmbeddings(),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        sections = split_sections(doc)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    print(f"Split the documents into {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
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

output_path = "../03_eval/data_metrics/v20241219/ch0503_zh_section/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
