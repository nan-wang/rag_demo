import operator
from typing import Annotated

from langgraph.graph import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict


class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]


def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


graph_builder = StateGraph(OverallState)
graph_builder.add_node(
    "generate_joke", lambda state: {"jokes": [f"讲一个关于{state['subject']}的冷笑话"]}
)
graph_builder.add_conditional_edges(START, continue_to_jokes)
graph_builder.add_edge("generate_joke", END)
app = graph_builder.compile()

# Invoking with two subjects results in a generated joke for each
output = app.invoke({"subjects": ["汽车", "小猫"]})
print(output)