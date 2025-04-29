from operator import add
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


class State(TypedDict):
    input: str
    node_trace: Annotated[list[str], add]


def node_a(state: State):
    return {"node_trace": ["a"]}


def node_b(state: State):
    return {"node_trace": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"input": "foo"}, config)
print(list(graph.get_state_history(config)))