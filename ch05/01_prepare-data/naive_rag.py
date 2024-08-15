import glob

import dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o-2024-08-06")

# walk through the text files under "data" directory
docs = []
for file in glob.glob("data/*.txt"):
    loader = TextLoader(file)
    _docs = loader.load()
    docs += _docs

print(f"Loaded {len(docs)} documents")
# print(f"Sample doc {docs[0].page_content[:200]}")

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    add_start_index=True,
)

all_splits = text_splitter.split_documents(docs)

print(f"Split the documents into {len(all_splits)} chunks")
print(f"all_splits[0].page_content {all_splits[0].page_content}")
print(f"all_splits[0] {len(all_splits[0].page_content)}")
print(f"all_splits[0] {len(all_splits[0].metadata)}")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke("中华人民共和国第一次参加冬奥会是哪一年?")

# print(f"Retrieved {len(retrieved_docs)} documents")
# print(f"Retrieved docs {retrieved_docs}")

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
