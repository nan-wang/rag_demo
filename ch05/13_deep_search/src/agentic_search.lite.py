import json
import operator
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Dict, Union, List

import dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class Configuration(BaseModel):
    max_web_search_loops: int = Field(
        default=3,
        title="Search Depth",
        description="Number of search iterations to perform"
    )
    reasoning_llm: str = Field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        title="LLM Model Name",
        description="Name of the LLM model to use"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.configurable if hasattr(config, 'configurable') and config else {}

        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)


summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user query from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                              
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user query, skip it.                                                            
4. Ensure all additions are relevant to the user's query.                                                         
5. Verify that your final output differs from the input summary.
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary to ask the user's query: {user_query}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for search from a collection of Wikipedia pages. 

</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}

If you don't find any knowledge gaps, just say "No knowledge gaps found." in the knowledge_gap and return an empty string in the follow_up_query.

MUST RETURN the ``knowledge_gap`` and ``follow_up_query`` in Chinese!!!
</Task>

Provide your analysis in JSON format:"""


@dataclass(kw_only=True)
class SummaryState:
    user_query: str = field(default=None)  # original user query
    search_query: str = field(default=None)  # search query
    web_search_results: Annotated[list, operator.add] = field(default_factory=list)  # web search results
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    search_loop_count: int = field(default=0)  # search loop count
    running_summary: str = field(default=None)  # final report


@dataclass(kw_only=True)
class SummaryStateInput:
    user_query: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)


def strip_thinking_tokens(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def format_sources(search_results: Dict[str, Any]) -> str:
    if not search_results:
        return ""
    return "\n".join(
        f"* {source['title']}: {source['url']}" for source in search_results["results"]
    )


def extract_json_from_markdown(input_str):
    match = re.search(r'```json\s*(\{.*?\})\s*```', input_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    return input_str


dotenv.load_dotenv()

vector_db_dir = '../../data_chroma_jina_embeddings'
collection_name = 'olympic_games'
vectorstore = Chroma(
    persist_directory=vector_db_dir,
    embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
    create_collection_if_not_exists=False,
    collection_name=collection_name)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def deduplicate_and_format_sources(
        search_response: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Invalid search response format")

    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
    return formatted_text.strip()


def search(state: SummaryState):
    if state.search_query is None:
        state.search_query = state.user_query
    results = retriever.invoke(f"{state.search_query}")
    search_results = {"results": []}
    for idx, doc in enumerate(results):
        content = doc.page_content.replace("\n", " ")
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        search_results["results"].append({
            "url": url,
            "content": content,
            "title": title
        })
    search_str = deduplicate_and_format_sources(search_results)
    return {
        "sources_gathered": [format_sources(search_results)],
        "search_loop_count": state.search_loop_count + 1,
        "web_search_results": [search_str, ]
    }


def summarize_sources(state: SummaryState, config: RunnableConfig):
    existing_summary = state.running_summary

    most_recent_web_search = state.web_search_results[-1]

    if existing_summary:
        human_message_content = (
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary> \n\n"
            f"<New Context> \n {most_recent_web_search} \n <New Context> \n\n"
            f"Update the Existing Summary with the New Context on this topic: \n "
            f"<User Input> \n {state.user_query} \n <User Input> \n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_search} \n <Context> \n\n"
            f"Create a Summary using the Context to answer the user query: \n "
            f"<User Input> \n {state.user_query} \n <User Input> \n\n"
        )

    configurable = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)

    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions), HumanMessage(content=human_message_content)]
    )
    running_summary = result.content

    if configurable.strip_thinking_tokens:
        running_summary = strip_thinking_tokens(running_summary)
    return {"running_summary": running_summary}


def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)
    result = llm.invoke(
        [SystemMessage(content=reflection_instructions.format(user_query=state.user_query)),
         HumanMessage(content=(
             f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n "
             f"And now identify a knowledge gap and generate a follow-up search query:"))]
    )
    content = result.content
    try:
        if content.startswith("```json"):
            content = extract_json_from_markdown(content)
        reflection_content = json.loads(content)
        query = reflection_content.get("follow_up_query")
        if not query:
            return {"search_query": f"{state.user_query}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {"search_query": f"{state.user_query}"}


def finalize_summary(state: SummaryState):
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        for line in source.split("\n"):
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources = "\n".join(unique_sources)
    state.running_summary = f"## Answer\n{state.running_summary}\n\n## Sources\n{all_sources}"
    return {"running_summary": state.running_summary}


def route_search(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "search"]:
    configurable = Configuration.from_runnable_config(config)
    if not state.search_query:
        return "finalize_summary"
    if state.search_loop_count < configurable.max_web_search_loops:
        return "search"
    else:
        return "finalize_summary"


builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("search", search)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

builder.add_edge(START, "search")
builder.add_edge("search", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_search)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
output = graph.invoke({"user_query": "2014年冬奥会的吉祥物是什么?是如何被选出的？"})
print(output["running_summary"])
