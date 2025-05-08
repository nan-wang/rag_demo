import os
from pprint import pprint

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


class State(TypedDict):
    question: str
    decomposed_questions: List[str]
    context: List[Document]
    answer: str


vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=JinaEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL")),
    create_collection_if_not_exists=False,
    collection_name=COLLECTION_NAME)

prompt_template = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. 
It is May 1st, 2025 today.
Question: {question} 
Context: {context} 
Answer:
""")

decomposition_prompt_template = ChatPromptTemplate.from_template("""
你是一名国际奥林匹克委员会的专家。你擅长分析关于奥运会的问题。你的任务是判断用户提出的问题是否需要拆解为多个子问题，每个子问题可以被独立回答。今天的日期是2025年5月1日。
必须满足以下要求:
- 每个问题可以被独立回答
- 子问题必须是中文
- 不要做任何解释或者输出任何其他无关信息！
- 直接输出子问题

例子1：
用户问题：列举过去5届夏季奥运会吉祥物的名称?
子问题: 
2024年夏季奥运会吉祥物名称是什么？
2020年夏季奥运会吉祥物名称是什么？
2016年夏季奥运会吉祥物名称是什么？
2012年夏季奥运会吉祥物名称是什么？
2008年夏季奥运会吉祥物名称是什么？

用户问题: {question}
Output:""")

llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0)
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4})

def decompose_query(state: State):
    prompt = decomposition_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response_message = llm.invoke(prompt)
    question_list = response_message.content.split("\n")
    return {"decomposed_questions": question_list}


def retrieve(state: State):
    retrieved_docs = retriever.batch(state["decomposed_questions"])
    results = []
    for l in retrieved_docs:
        results.extend(l)
    return {"context": results}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke(
        {
            "question": state["question"], "context": docs_content
        }
    )
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


graph_builder = StateGraph(State)

graph_builder.add_node(decompose_query)
graph_builder.add_node(retrieve)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "decompose_query")
graph_builder.add_edge( "decompose_query", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

query = "过去5届夏季奥运会都是在哪里举办的?"
result = graph.invoke({"question": query})
pprint(result["decomposed_questions"])

# 输出
# [
#   '2024年夏季奥运会举办地是哪里？',
#   '2020年夏季奥运会举办地是哪里？',
#   '2016年夏季奥运会举办地是哪里？',
#   '2012年夏季奥运会举办地是哪里？',
#   '2008年夏季奥运会举办地是哪里？'
#   ]
