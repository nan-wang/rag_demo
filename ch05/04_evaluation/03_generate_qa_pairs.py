import asyncio
import argparse
import json
import random
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from synthetic_data_prompt import SYSTEM_PROMPT, USER_PROMPT
from loguru import logger

dotenv.load_dotenv()


class QAPair(BaseModel):
    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The answer to the question.")


class State(TypedDict):
    length: int
    clarity: str
    difficulty: str
    document_id: str
    context: str
    query: str
    answer: str


def select_length(state: State):
    return {"length": random.choice([8, 16, 32])}


def select_clarity(state: State):
    return {"clarity": random.choice(["简单", "基础", "困难"])}


def select_difficulty(state: State):
    return {"difficulty": random.choice(["小学", "初中", "高中", "大学", "研究生博士"])}


def generate(state: State):
    prompt = prompt_template.invoke(
        {
            "context_str": state["context"],
            "length": state["length"],
            "clarity": state["clarity"],
            "difficulty": state["difficulty"],
        }
    )
    qa_pair = llm.invoke(prompt)
    return {
        "query": qa_pair.question,
        "answer": qa_pair.answer,
    }


QUESTION_GEN_SYS_TMPL = (
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

QUESTION_GEN_USER_TMPL = (
    HumanMessagePromptTemplate.from_template(USER_PROMPT))

prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_GEN_SYS_TMPL,
        QUESTION_GEN_USER_TMPL
    ]
)

llm = ChatOpenAI(model="Qwen/Qwen3-8B").with_structured_output(QAPair)


async def main(num_docs: int, index_dir: str, collection_name: str, output_dir: str, max_concurrency: int=8):
    graph_builder = StateGraph(State)
    graph_builder.add_node(select_length)
    graph_builder.add_node(select_clarity)
    graph_builder.add_node(select_difficulty)
    graph_builder.add_node(generate)

    graph_builder.add_edge(START, "select_length")
    graph_builder.add_edge(START, "select_clarity")
    graph_builder.add_edge(START, "select_difficulty")
    graph_builder.add_edge("select_length", "generate")
    graph_builder.add_edge("select_clarity", "generate")
    graph_builder.add_edge("select_difficulty", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    vectorstore = Chroma(persist_directory=index_dir, collection_name=collection_name)
    ids = vectorstore.get()['ids']
    logger.info(f"Total number of documents: {len(ids)}")
    random.shuffle(ids)
    num_docs = min(num_docs, len(ids))
    logger.info(f"Generating {num_docs} documents...")
    selected_docs = \
        {k: v for k, v in vectorstore.get(ids=ids[:num_docs]).items() if k in ("ids", "metadatas", "documents")}
    selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

    inputs = [
        {
            "document_id": doc["ids"],
            "context": doc["documents"],
        } for doc in selected_docs
    ]

    qa_pairs = []
    logger.info(f"Total number of documents: {len(inputs)}")

    outputs = await graph.abatch(inputs, config=RunnableConfig(max_concurrency=max_concurrency))
    logger.info(f"Complete generation")
    for result in outputs:
        qa_doc = {
            "query": result["query"],
            "ground_truth": {
                "contexts": [result["context"], ],
                "content": result["answer"]
            },
            "metadatas": {
                "length": result["length"],
                "clarity": result["clarity"],
                "difficulty": result["difficulty"],
                "document_id": result["document_id"],
            }
        }
        qa_pairs.append(qa_doc)

    output_path = f"{output_dir}/qa_pairs.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(qa_pairs)} pairs to {output_path}")


if __name__ == "__main__":
    # python 03_generate_qa_pairs.py --index_dir ./data_chroma_multi  --collection_name test_db --output_dir data_eval/v20250501 --num_docs 64
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, help="Number of documents to generate.")
    parser.add_argument("--index_dir", type=str, help="Directory where the index is stored.")
    parser.add_argument("--collection_name", type=str, help="Name of the collection.")
    parser.add_argument("--output_dir", type=str, help="Directory where the output is stored.")
    parser.add_argument("--max_concurrency", type=int, help="Maximum number of concurrent runs.")
    args = parser.parse_args()
    asyncio.run(
        main(
            num_docs=args.num_docs,
            index_dir=args.index_dir,
            collection_name=args.collection_name,
            output_dir=args.output_dir
        )
    )
