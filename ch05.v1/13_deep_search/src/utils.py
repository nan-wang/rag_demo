from typing import Any, Dict, List, Union
from tavily import TavilyClient
import re
import json


def get_config_value(value: Any) -> str:
    return value if isinstance(value, str) else value.value


def strip_thinking_tokens(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def tavily_search(query: str, fetch_full_page: bool = True, max_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    client = TavilyClient()
    return client.search(
        query,
        max_results=max_results,
        include_full_page=fetch_full_page)


def deduplicate_and_format_sources(
        search_response: Union[Dict[str, Any], List[Dict[str, Any]]],
        max_tokens_per_source: int = 1000,
        fetch_full_page: bool = True) -> str:
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Invalid search response format")

    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
    return formatted_text.strip()


def format_sources(search_results: Dict[str, Any]) -> str:
    if not search_results:
        return ""
    return "\n".join(
        f"* {source['title']}: {source['url']}" for source in search_results["results"]
    )


def extract_json_from_markdown(input_str):
    match = re.search(r'```json\s*(\{.*?\})\s*```', input_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    return input_str


import glob
import re
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_year(fn):
    return int(Path(fn).stem[:4])


def get_season(fn):
    return Path(fn).stem[5:7]


def load_documents(pathname: str, with_metadata=False):
    docs = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        _docs = loader.load()
        if with_metadata:
            year = get_year(file)
            season = get_season(file)
            for doc in _docs:
                doc.metadata["year"] = year
                doc.metadata["season"] = season
                docs.append(doc)
        else:
            docs += _docs
    return docs


def format_docs(docs):
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx + 1}]{doc_str}")
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


