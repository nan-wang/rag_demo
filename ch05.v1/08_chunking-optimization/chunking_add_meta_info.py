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
from typing import Iterable

import dotenv

dotenv.load_dotenv()


def format_docs(docs):
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx+1}]{doc_str}")
    return "\n\n".join(output_list)


def split_sections(text, source=None, skip_empty_sections=False):
    sections = []
    pattern = r'(==+)(.*?)==+\s*([^=]*)'

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    parent_title = ""
    prev_level = 0
    section_title = ["", ]
    text = f"== summary ==\n\n{text}"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        level = len(match.group(1)) - 1  # Determine the section level by the number of '='
        title = match.group(2).strip()
        content = match.group(3).strip()

        if prev_level == 0:
            section_title.append(title)
            prev_level = level
        else:
            if prev_level == level:
                # pop the last section title
                section_title.pop()
                # push the current section title
                section_title.append(title)
            elif prev_level < level:
                # set the parent section title
                section_title.append(title)
                prev_level = level
            elif prev_level > level:
                section_title.pop()
                for _ in range(prev_level - level):
                    section_title.pop()
                section_title.append(title)
                prev_level = level
        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
        if level == 2:
            section_counters[3] = -1
        section_counters[level] += 1

        metadata = {
            "source": source,
            "title": title,
            "parent_section": "_".join(section_title[1:-1]),
            "section_level": level,
            "section_index": section_counters[level]
        }
        if title in ["注释", "参见", "参考文献", "外部链接", "奖牌榜", "比赛日程", "参考"]:
            continue
        if skip_empty_sections and not content:
            continue
        sections.append(Document(page_content=content, metadata=metadata))
    return sections


def split_chunks(docs: Iterable[Document]):
    results = []
    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
        separators=['。', '！', '？', '\?', '\n\n', '\n', '\n\n\n'],
        is_separator_regex=True,
        keep_separator="end"
    )

    for chunk in text_splitter.split_documents(docs):
        content = chunk.page_content
        metadata = chunk.metadata
        section_title = metadata["title"]
        if metadata["parent_section"]:
            section_title = f"{metadata['parent_section']}_{section_title}"
        content = f"section_title: {section_title}\ncontent: {content}"
        article_title = Path(metadata.get("source", "")).name.removesuffix(".txt")
        content = f"article_title: {article_title}\n{content}"
        results.append(Document(page_content=content, metadata=metadata))

    return results


vector_db_dir = '../data_chroma_add_meta_info'
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
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
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

output_path = "../03_eval/data_metrics/v20241219/ch0503_add_meta_info/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
