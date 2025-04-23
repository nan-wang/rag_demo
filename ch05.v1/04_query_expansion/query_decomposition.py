from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_documents, get_chunks

dotenv.load_dotenv()

vector_db_dir = '../data_chroma'
collection_name = 'test_db'

docs = load_documents("../data/*.txt")
print(f"Loaded {len(docs)} documents")

chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")

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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

from langchain.prompts import ChatPromptTemplate

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
The sub-problems should be answerable by a collection of documents that contains the information about the olympic games from 1980 to 2024. \n
Generate multiple search queries related to: {question} \n
Output:"""

prompt_decomposition = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-2024-08-06")

generate_queries_decomposition = (
        prompt_decomposition
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

query = "列举过去5届夏季奥运会吉祥物的名称?"

retrieval_chain = generate_queries_decomposition | retriever.map()

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


def format_docs(retrieval_docs_from_multi_query):
    docs = []
    for r in retrieval_docs_from_multi_query:
        docs += r
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
        {"context": retrieval_chain | format_docs, "question": RunnablePassthrough()}
        | RunnableParallel(
    contexts=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

result = rag_chain.invoke(query)
print(result)

# 'question': '列举过去5届夏季奥运会吉祥物的名称?',
# 'answer':
# '1. 2024年巴黎奥运会 - 奥林匹克弗里热（Phryges）\n'
# '2. 2020年东京奥运会 - 未提供\n'
# '3. 2016年里约热内卢奥运会 - 费尼希斯（Vinicius）\n'
# '4. 2012年伦敦奥运会 - 文洛克（Wenlock）\n'
# '5. 2008年北京奥运会 - 福娃（贝贝、晶晶、欢欢、迎迎、妮妮）'}
