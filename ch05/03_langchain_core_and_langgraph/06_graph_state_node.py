from langgraph.graph import StateGraph, START

from typing_extensions import TypedDict


class State(TypedDict):
    step_counter: int
    step_list: list[str]


def node_1(state: State):
    return {"step_counter": state.get("step_counter", 0) + 1, "step_list": state.get("step_list", []) + [f"node_1"]}


def node_2(state: State):
    return {"step_counter": state.get("step_counter", 0) + 1, "step_list": state.get("step_list", []) + [f"node_2"]}


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_edge(START, "node_1")
graph_builder.add_edge("node_1", "node_2")
app = graph_builder.compile()
output = app.invoke({"step_counter": 0, "step_list": []})
print(output)