from operator import add
from typing import Annotated

from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict


class State(TypedDict):
    msg_list: Annotated[list[str], add]


def node_1(state: State):
    return {"msg_list": ["hello,"]}


def node_2(state: State):
    return {"msg_list": ["world!"]}


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
app = graph_builder.compile()
output = app.invoke({"msg_list": []})
print(output)