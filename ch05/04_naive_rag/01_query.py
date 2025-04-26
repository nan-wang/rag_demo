import dotenv
from pprint import pprint
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

VECTOR_DB_DIR = "../index_chroma_naive"
COLLECTION_NAME = "olympic_games"

dotenv.load_dotenv()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


vector_store = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        chunk_size=16,
        check_embedding_ctx_length=False
    ),
    create_collection_if_not_exists=False,
    collection_name=COLLECTION_NAME)

prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
)


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct")


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"question": state["question"], "context": docs_content})
    response_message = llm.invoke(prompt)
    return {"answer": response_message.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

query = "2024年巴黎奥运会的开幕式是哪一天?"
result = graph.invoke({"question": query})
pprint(result)

# 输出
# {
#   "answer": "2024年巴黎奥运会的开幕式于2024年7月26日晚上7点30分举行。",
#   "context": [
#     {
#       "id": "b6d9...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 2567
#       },
#       "page_content": "开幕式于欧洲中部时间2024年7月26日晚..."
#     },
#     {
#       "id": "768b...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 0
#       },
#       "page_content": "2024年夏季奥林匹克运动会 （英语..."
#     },
#     {
#       "id": "dfa1...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 5722
#       },
#       "page_content": "== 赞助商 ==..."
#     },
#     {
#       "id": "94fa...",
#       "metadata": {
#         "source": "2024年夏季奥林匹克运动会.txt",
#         "start_index": 2122
#       },
#       "page_content": "=== 火炬 ===..."
#     }
#   ],
#   "question": "2024年巴黎奥运会的开幕式是哪一天?"
# }