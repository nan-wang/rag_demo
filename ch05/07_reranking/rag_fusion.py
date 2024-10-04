from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_documents, get_chunks, format_docs

dotenv.load_dotenv()

vector_db_dir = '../data_chroma'
collection_name = 'test_db'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir,
        collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

from langchain.prompts import ChatPromptTemplate

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-2024-08-06")

generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain.retrievers import ContextualCompressionRetriever

compressor = JinaRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

retrieval_chain_rag_fusion = generate_queries | compression_retriever

# result = generate_queries.invoke("奥运会的奖牌有什么环保设计?")

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    RunnableParallel(
        context=retrieval_chain_rag_fusion | format_docs,
        question=RunnablePassthrough())
    | RunnableParallel(
        contexts=RunnablePick("context"),
        question=RunnablePick("question"),
        answer=prompt | llm | StrOutputParser())
)

query = "奥运会的奖牌有什么环保设计?"
result = rag_chain.invoke(query)
print(result['contexts'])