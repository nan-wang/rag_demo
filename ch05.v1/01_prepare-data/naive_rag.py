from pathlib import Path

import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_documents, get_chunks, format_docs

dotenv.load_dotenv()

VECTOR_DB_DIR = '../data_chroma_test_db'
COLLECTION_NAME = 'olympic_games'


if Path(VECTOR_DB_DIR).exists():
    vectorstore = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=OpenAIEmbeddings(),
        create_collection_if_not_exists=False,
        collection_name=COLLECTION_NAME)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


llm = ChatOpenAI(model="gpt-4o-2024-08-06")
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
)

query = "2024年巴黎奥运会的开幕式是哪一天?"
result = rag_chain.invoke(query)
print(result)
