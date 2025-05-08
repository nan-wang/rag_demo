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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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