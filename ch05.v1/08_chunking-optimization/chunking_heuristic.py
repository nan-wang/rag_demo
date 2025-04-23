from utils import load_documents, get_chunks
from langchain_community.document_loaders import TextLoader


loader = TextLoader("../data/2008年夏季奥林匹克运动会.txt")
docs = loader.load()
# split by section
# use recursive character text splitter to split the sections into chunks
chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")
for i, split in enumerate(chunks):
    print(f"Split {i} {repr(split.page_content)}")
