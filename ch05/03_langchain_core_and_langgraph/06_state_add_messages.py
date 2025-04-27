from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    msg_list: Annotated[list[AnyMessage], add_messages]


def node_1(state: State):
    return {"msg_list": [{"type": "human", "content": "你好！"}]}


def node_2(state: State):
    return {"msg_list": [{"type": "assistant", "content": "你好，有什么可以帮助您的?"}]}


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
app = graph_builder.compile()
output = app.invoke({"msg_list": []})
print(output)