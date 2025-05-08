import os
from pprint import pprint

import dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

dotenv.load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


class State(TypedDict):
    question: str
    stepback_question: str
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

examples = [
    {
        "input": "巴黎奥运会的吉祥物有什么含义?",
        "output": "介绍巴黎奥运会的吉祥物",
    },
    {
        "input": "2004年奥运会和雅典共同竞争举办权的有哪些国家?",
        "output": "介绍2024年夏季奥运会的申办过程",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
template = """You are an expert at Olympic Games.\n
Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.\n
The generic step-back question should be answerable given a collection of wikipedia pages that contains \n
the information about the olympic games from 1980 to 2024.\n
Here are a few examples:
"""
stepback_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

llm = ChatOpenAI(model="Qwen/Qwen3-32B", temperature=0)
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4})

def stepback(state: State):
    prompt = stepback_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response_message = llm.invoke(prompt)
    return {"stepback_question": response_message.content}


def retrieve(state: State):
    retrieved_docs = retriever.batch([state["question"], state["stepback_question"]])
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

graph_builder.add_node(stepback)
graph_builder.add_node(retrieve)
graph_builder.add_node(generate)

graph_builder.add_edge(START, "stepback")
graph_builder.add_edge( "stepback", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

query = "北京申办过几次奥运会?"
result = graph.invoke({"question": query})
pprint(result["stepback_question"])

# 输出:
# '介绍北京申办奥运会的历史'
