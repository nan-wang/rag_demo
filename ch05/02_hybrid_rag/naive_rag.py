import glob

import dotenv
from pathlib import Path

import pkuseg
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


def get_all_splits():
    docs = []
    for file in glob.glob("data/*.txt"):
        loader = TextLoader(file)
        _docs = loader.load()
        docs += _docs

    print(f"Loaded {len(docs)} documents")

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )

    return text_splitter.split_documents(docs)


vector_db_dir = 'data_chroma'
collection_name = 'test_db'
if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"{vectorstore._chroma_collection.count()} documents loaded")
else:
    # walk through the text files under "data" directory
    all_splits = get_all_splits()
    print(f"Split the documents into {len(all_splits)} chunks")
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir,
        collection_name=collection_name)

vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# set up the bm25 retriever
seg = pkuseg.pkuseg()


def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != '']
        result += split_tokens
    return result


all_splits = get_all_splits()
bm25_retriever = BM25Retriever.from_documents(all_splits, preprocess_func=tokenize_doc)
bm25_retriever.k = 10

from langchain.retrievers import EnsembleRetriever

retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

# retrieved_docs = retriever.invoke("奥运会金牌的挂袋有哪些设计?")

# print(f"Retrieved {len(retrieved_docs)} documents")
# for d in retrieved_docs:
#     print(f"Retrieved doc, {repr(d.page_content)}")
#     print(f"Retrieved doc meta, {d.metadata}")
#
# sys.exit(0)

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

query = "哪一届奥运会第一次发现了兴奋剂？"

result = rag_chain.invoke(query)
print(result)
