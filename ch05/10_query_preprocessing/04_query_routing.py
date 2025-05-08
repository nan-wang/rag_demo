from typing import Literal
from pprint import pprint

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START

dotenv.load_dotenv(".oai.env")

class State(TypedDict):
    question: str
    context: str
    answer: str
    datasource: str


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["hosts", "medals", "general_description"] = Field(
        ...,
        description="Given a user query, route it to the most relevant datasource for answering their question",
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to the appropriate datasource.
Based on the information needed to answer the user's question, you route the question to the most relevant datasource.
The default datasource is `general_description` which contains wiki pages about the Olympics from 1980 to 2024.
"""

route_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

def route(state: State):
    prompt = route_prompt_template.invoke(
        {
            "question": state["question"],
        }
    )
    response = llm.invoke(prompt)
    return {
        "datasource": response.datasource,
    }


graph_builder = StateGraph(State)
graph_builder.add_node(route)
graph_builder.add_edge(START, "route")
graph = graph_builder.compile()

query = "里约奥运会哪个国家获得的金牌最多?"
result = graph.invoke({"question": query})
pprint(result)
# 输出：
# {'datasource': 'medals', 'question': '里约奥运会哪个国家获得的金牌最多?'}
