import sys
import json

import dotenv
from pathlib import Path
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_documents, get_chunks, format_docs, split_contexts

dotenv.load_dotenv()

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["hosts", "medals", "general_description"] = Field(
        ...,
        description="Given a user query, route it to the most relevant datasource for answering their question",
    )

llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

structure_llm = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate datasource.
Based on the information needed to answer the user's question, you route the question to the most relevant datasource.
The default datasource is `general_description` which contains wiki pages about the Olympics from 1980 to 2024.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

router = prompt | structure_llm

query = "里约奥运会哪个国家获得的金牌最多?"
result = router.invoke({"question": query})
print(result)
exit(0)

vector_db_dir = '../data_chroma'
collection_name = 'test_db'


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

query = "介绍北京申办奥运会的历史"
# retrieved_docs = retriever.invoke(query)

# print(f"Retrieved {len(retrieved_docs)} documents")
# for doc in retrieved_docs:
#     print(f"Retrieved doc, {repr(doc.page_content[:100])}")
#     print(f"Retrieved doc meta, {doc.metadata}")
# exit(0)


llm = ChatOpenAI(model="gpt-4o-2024-08-06")
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableParallel(
            contexts=RunnablePick("context"),
            question=RunnablePick("question"),
            answer=prompt | llm | StrOutputParser())
)


result = rag_chain.invoke(query)
