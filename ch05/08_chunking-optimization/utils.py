import re
import copy
from pathlib import Path
import glob


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from collections import defaultdict


def build_hierarchy(data):
    hierarchy = {}
    sections_by_title = {}

    # Step 1: Create initial dictionary with sections at top-level
    for item in data:
        title = item['title']
        section_level = item['section_level']
        parent_section = item['parent_section']

        # Add item to the dictionary
        item_copy = {
            'title': title,
            'content': item['content'],
            'subsections': [],
            'section_index': item['section_index'],
            'section_level': item['section_level'],
        }

        if section_level == 0:
            # Top-level sections go directly into hierarchy
            hierarchy[title] = item_copy
        else:
            # Other levels will be stored in sections_by_title
            sections_by_title[title] = item_copy

    # Step 2: Assign subsections to their parent sections
    for item in data:
        if item['section_level'] > 0:
            title = item['title']
            parent_title = item['parent_section']

            # Find the parent section and add current section as a subsection
            if parent_title in hierarchy:
                parent_section = hierarchy[parent_title]
            elif parent_title in sections_by_title:
                parent_section = sections_by_title[parent_title]
            else:
                continue

            parent_section['subsections'].append(sections_by_title[title])

    return [v for v in hierarchy.values()]


def split_sections(content, skip_empty_sections=True, root_title=""):
    sections = defaultdict(list)
    pattern = r'(==+)(.*?)==+\s*([^=]*)'

    # This dictionary helps to track the current section level and index
    section_counters = {1: -1, 2: -1, 3: -1}
    sections[0].append(
        {
            "title": root_title,
            "content": "",
            "parent_section": "",
            "section_level": 0,
            "section_index": 0
        }
    )
    text = f"== summary ==\n\n{content}"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        level = len(match.group(1)) - 1  # Determine the section level by the number of '='
        title = match.group(2).strip()
        content = match.group(3).strip()

        # Reset section index for the lower level when we encounter a higher-level section
        if level == 1:
            section_counters[2] = -1
            section_counters[3] = -1
        elif level == 2:
            section_counters[3] = -1
        parent_section = sections[level-1][-1]["title"]  # The parent section is the last level 1 section
        section_counters[level] += 1

        section_info = {
            "title": title,
            "content": content,
            "parent_section": parent_section,
            "section_level": level,
            "section_index": section_counters[level]
        }

        if title in ["注释", "参见", "参考文献", "外部链接", "奖牌榜", "比赛日程", "参考", "外部连结"]:
            continue
        sections[level].append(section_info)

    result = []
    # merge the sections
    for section in sections.values():
        # if section[0]["section_level"] == 0:
        #     continue
        result += section
    return build_hierarchy(result)


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



def get_chunks_at_multi_levels(docs):
    section_docs = []
    for doc in docs:
        section_list = split_sections(doc.page_content)
        section_docs += create_section_documents(section_list, doc.metadata, False, False)


def get_chunks(docs: list[Document]):
    # split each document into sections
    section_docs = []
    for doc in docs:
        section_list = split_sections(doc.page_content)
        section_docs += create_section_documents(section_list, doc.metadata, False, False)

    # split the sections into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )

    _chunks = text_splitter.split_documents(section_docs)

    chunks = []
    for doc in _chunks:
        metadata = doc.metadata
        section_title = metadata["section_title"]
        if metadata["parent_section"]:
            section_title = f"{metadata['parent_section']}_{section_title}"
        if not doc.page_content.startswith("article_title:"):
            content = f"section_title: {section_title}\ncontent: {doc.page_content}"
            title = Path(metadata.get("source", "")).name.removesuffix(".txt")
            content = f"article_title: {title}\n{content}"
            doc.page_content = content
            print(f"Updated content: {doc.page_content}")
        chunks.append(doc)
    return chunks


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
    return "\n\n".join(doc.page_content for doc in docs)


def split_contexts(contexts):
    pattern = r'(?=article_title:)'
    # Split the text using the regex pattern
    return [part.strip() for part in re.split(pattern, contexts) if part.strip()]


def flatten_sections(hierarchy):
    flattened_list = []

    def flatten_helper(section, parents, content_list):
        content = section['content']
        level = section['section_level']
        index = section['section_index']
        title = section['title']
        title_md = f"{'#' * (level + 1)} {title}"
        current_content_list = [f'{title_md}', ]
        if content:
            current_content_list.append(content)
        for subsection in section['subsections']:
            current_content_list += flatten_helper(subsection, parents + [title,], content_list + [f'{title_md}', ])
        content_list += current_content_list
        if section['subsections']:
            content = "\n".join(content_list)
        flattened_section = {
            'title': title,
            'level': level,
            'index': index,
            'content': content,
            'parents': parents,
            'is_leaf': not section['subsections'],
        }
        flattened_list.append(flattened_section)
        return current_content_list

    for node in hierarchy:
        flatten_helper(node, [], [])

    return flattened_list


def convert_chunks_to_documents(chunks, add_year=False, add_season=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=32,
        add_start_index=True,
        separators=['。', '！', '？', '\?', '\n\n', '\n', '\n\n\n'],
        is_separator_regex=True,
        keep_separator="end"
    )
    # split the leaf chunks into sentences
    _chunks = []
    for chunk in chunks:
        if not chunk["is_leaf"]:
            _chunks.append(chunk)
            continue
        content = chunk["content"]
        if not content:
            _chunks.append(chunk)
            continue
        # _chunks.append(chunk)
        sentences = text_splitter.split_text(content)
        if len(sentences) == 1:
            _chunks.append(chunk)
            continue
        chunk["is_leaf"] = False
        # add title to the chunk content
        content = []
        for i, parent in enumerate(chunk["parents"]):
            content.append(f"{'#'*(i+1)} {parent}")
        content.append(f"{'#'*(len(chunk['parents'])+1)} {chunk['title']}")
        content.append(chunk["content"])
        chunk["content"] = "\n".join(content)
        _chunks.append(chunk)
        for i, sentence in enumerate(sentences):
            _chunks.append({
                "content": sentence,
                "title": chunk["title"],
                "level": chunk["level"] + 1,
                "index": i,
                "parents": chunk["parents"],
                "is_leaf": True,
            })

    # construct documents
    for chunk in _chunks:
        if chunk["level"] == 0:
            continue
        if add_year:
            chunk["year"] = get_year(chunk["parents"][0])
        if add_season:
            chunk["season"] = get_season(chunk["parents"][0])
        content = []
        if chunk["is_leaf"]:
            for i, parent in enumerate(chunk["parents"]):
                content.append(f"{'#'*(i+1)} {parent}")
            content.append(f"{'#'*(len(chunk['parents'])+1)} {chunk['title']}")
        content.append(chunk["content"])
        metadata = {
            "section_title": chunk["title"],
            "section_level": chunk["level"],
            "section_index": chunk["index"],
            "parent_sections": "\n".join(chunk["parents"]),
            "is_leaf": chunk["is_leaf"],
        }
        if add_year:
            metadata["year"] = chunk["year"]
        if add_season:
            metadata["season"] = chunk["season"]
        yield Document(page_content="\n".join(content), metadata=metadata)


