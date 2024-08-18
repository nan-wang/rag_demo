import re
import copy
from pathlib import Path
import glob


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_sections(content, skip_empty_sections=True):
    sections = []
    pattern = r'(==+)(.*?)==+\s*([^=]*)'

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    parent_section = ""

    text = f"== summary ==\n\n{content}"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        level = len(match.group(1)) - 1  # Determine the section level by the number of '='
        title = match.group(2).strip()
        content = match.group(3).strip()

        if skip_empty_sections and not content:
            continue
        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
            parent_section = ""

        section_counters[level] += 1

        if level == 2:
            parent_section = sections[-1]["title"]  # The parent section is the last level 1 section

        section_info = {
            "title": title,
            "content": content,
            "parent_section": parent_section,
            "section_level": level,
            "section_index": section_counters[level]
        }

        if title in ["注释", "参见", "参考文献", "外部链接", "奖牌榜", "比赛日程", "参考"]:
            continue
        sections.append(section_info)

    return sections


def create_section_documents(section_list, metadatas, add_section_title=True, add_article_title=True):
    _metadatas = metadatas
    documents = []
    for i, sec in enumerate(section_list):
        metadata = copy.deepcopy(_metadatas)
        metadata["section_index"] = i
        metadata["section_level"] = sec["section_level"]
        metadata["parent_section"] = sec["parent_section"]
        metadata["section_title"] = sec["title"]
        content = sec["content"]
        section_title = metadata["section_title"]
        if metadata["parent_section"]:
            section_title = f"{metadata['parent_section']}_{section_title}"
        if add_section_title:
            content = f"section_title: {section_title}\ncontent: {content}"
        if add_article_title:
            title = Path(metadata.get("source", "")).name.removesuffix(".txt")
            content = f"article_title: {title}\n{content}"
        new_doc = Document(page_content=content, metadata=metadata)
        documents.append(new_doc)
    return documents


def get_chunks(docs: list[Document]):
    # split each document into sections
    section_docs = []
    for doc in docs:
        section_list = split_sections(doc.page_content)
        section_docs += create_section_documents(section_list, doc.metadata)

    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )

    return text_splitter.split_documents(section_docs)


def load_documents(pathname: str):
    docs = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        _docs = loader.load()
        docs += _docs
    return docs


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
