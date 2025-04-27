from langgraph.graph import StateGraph, START

from typing_extensions import TypedDict


class State(TypedDict):
    input: int
    msg: str


def node_1(state: State):
    return {"msg": state.get("msg", "") + "node_1"}


def node_2(state: State):
    return {"msg": state.get("msg", "") + "node_2"}


def routing_function(state: State):
    # 通过输入的input值来决定下一个节点
    if state["input"] >= 0:
        return "node_1"
    else:
        return "node_2"


graph_builder = StateGraph(state_schema=State)
graph_builder.add_node(node_1)
graph_builder.add_node(node_2)
graph_builder.add_conditional_edges(START, routing_function)

app = graph_builder.compile()
print(app.invoke({"input": 1}))
# 输出：{'input': 1, 'msg': 'node_1'}
print(app.invoke({"input": -1}))
# 输出：{'input': -1, 'msg': 'node_2'}