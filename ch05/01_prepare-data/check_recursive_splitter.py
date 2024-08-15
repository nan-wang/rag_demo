from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=32,
    add_start_index=True,
)


docs = TextLoader("data/1984年夏季奥林匹克运动会.txt").load()
all_splits = text_splitter.split_documents(docs)

print(f"Split the documents into {len(all_splits)} chunks")

for i, split in enumerate(all_splits):
    print(f"Split {i} {repr(split.page_content)}")