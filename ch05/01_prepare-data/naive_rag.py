import glob
import sys

import dotenv
from pathlib import Path
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

vector_db_dir = 'data_chroma'
collection_name = 'test_db'

from langchain_core.documents import Document


def get_chunks(docs: list[Document]):
    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )

    return text_splitter.split_documents(docs)


def load_documents(pathname: str):
    docs = []
    for file in glob.glob(pathname):
        loader = TextLoader(file)
        _docs = loader.load()
        docs += _docs
    return docs


if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(), create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir, collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("中华人民共和国第一次参加冬奥会是哪一年?")

print(f"Retrieved {len(retrieved_docs)} documents")
print(f"Retrieved doc, {repr(retrieved_docs[0].page_content[:100])}")
print(f"Retrieved doc meta, {retrieved_docs[0].metadata}")

sys.exit(0)


llm = ChatOpenAI(model="gpt-4o-2024-08-06")
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

query = "中华人民共和国第一次参加冬奥会是哪一年?"

result = rag_chain.invoke(query)
print(result)
