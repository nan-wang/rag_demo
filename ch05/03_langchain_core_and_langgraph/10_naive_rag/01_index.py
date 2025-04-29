import os
import glob

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def load_documents(pathname: str):
    doc_list = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        doc_list += loader.load()
    return doc_list


def get_chunks(doc_list: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )
    return text_splitter.split_documents(doc_list)


docs = load_documents("../../data/*.txt")
print(f"Loaded {len(docs)} documents")

chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        chunk_size=16,
        check_embedding_ctx_length=False
    ),
    persist_directory=VECTOR_DB_DIR,
    collection_name=COLLECTION_NAME,
)