import json
from pathlib import Path
from typing import Literal

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from configuration import Configuration
from prompts import query_writer_instructions, get_current_date, summarizer_instructions, reflection_instructions
from state import SummaryState, SummaryStateInput, SummaryStateOutput
from utils import strip_thinking_tokens, get_config_value, tavily_search, format_sources, \
    deduplicate_and_format_sources, extract_json_from_markdown, load_documents, split_chunks, split_sections

dotenv.load_dotenv()


def get_all_splits():
    docs = load_documents("../../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


vector_db_dir = '../../data_chroma_jina_embeddings'
collection_name = 'olympic_games'
if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"{vectorstore._chroma_collection.count()} documents loaded")
vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

seg = pkuseg.pkuseg()

def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != '']
        result += split_tokens
    return result


chunks = get_all_splits()
bm25_retriever = BM25Retriever.from_documents(
    chunks, preprocess_func=tokenize_doc)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)


def generate_query(state: SummaryState, config: RunnableConfig):
    """Generate a search query based on the current state.

    :param state:
    :param config:
    :return:
    """
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        search_topic=state.search_topic
    )

    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model=configurable.reasoning_llm, temperature=0)

    result = llm.invoke(
        [SystemMessage(content=formatted_prompt), HumanMessage(content=f"Generate a query for web search:")]
    )

    content = result.content

    try:
        if content.startswith("```json"):
            content = extract_json_from_markdown(content)
        query = json.loads(content)
        search_query = query["query"]
    except (json.JSONDecodeError, KeyError):
        if configurable.strip_thinking_tokens:
            content = strip_thinking_tokens(content)
        search_query = content
    return {"search_query": search_query}


def search(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
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
    search_str = deduplicate_and_format_sources(
        search_results,
        fetch_full_page=False
    )
    return {
        "sources_gathered": [format_sources(search_results)],
        "search_loop_count": state.search_loop_count + 1,
        "web_search_results": [search_str, ]
    }


def web_search(state: SummaryState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_str = ""
    search_results = {}

    if search_api == "tavily":
        search_results = tavily_search(
            state.search_query,
            fetch_full_page=configurable.fetch_full_page,
            max_results=configurable.max_web_search_results
        )
        search_str = deduplicate_and_format_sources(
            search_results,
            max_tokens_per_source=1000,
            fetch_full_page=configurable.fetch_full_page
        )
    elif search_api == "duckduckgo":
        ...
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")
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
            f"<User Input> \n {state.search_topic} \n <User Input> \n\n"
        )
    else:
        human_message_content = (
            f"<Context> \n {most_recent_web_search} \n <Context> \n\n"
            f"Create a Summary using the Context on this topic: \n "
            f"<User Input> \n {state.search_topic} \n <User Input> \n\n"
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
        [SystemMessage(content=reflection_instructions.format(search_topic=state.search_topic)),
         HumanMessage(content=(
             f"Reflect on our existing knowledge: \n === \n {state.running_summary}, \n === \n "
             f"And now identify a knowledge gap and generate a follow-up web search query:"))]
    )
    content = result.content
    try:
        if content.startswith("```json"):
            content = extract_json_from_markdown(content)
        reflection_content = json.loads(content)
        query = reflection_content.get("follow_up_query")
        if not query:
            return {"search_query": f"Tell me more about {state.search_topic}"}
        return {"search_query": query}
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {"search_query": f"Tell me more about {state.search_topic}"}


def finalize_summary(state: SummaryState):
    seen_sources = set()
    unique_sources = []

    for source in state.sources_gathered:
        for line in source.split("\n"):
            if line.strip() and line not in seen_sources:
                seen_sources.add(line)
                unique_sources.append(line)

    all_sources = "\n".join(unique_sources)
    state.running_summary = f"## Summary\n{state.running_summary}\n\n## Sources\n{all_sources}"
    return {"running_summary": state.running_summary}


def route_search(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "search"]:
    configurable = Configuration.from_runnable_config(config)
    if state.search_loop_count < configurable.max_web_search_loops:
        return "search"
    else:
        return "finalize_summary"


builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("search", search)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "search")
builder.add_edge("search", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_search)
builder.add_edge("finalize_summary", END)

graph = builder.compile(debug=True)
output = graph.invoke({"search_topic": "北京申办过几次奥运会?"})
print(output["running_summary"])
