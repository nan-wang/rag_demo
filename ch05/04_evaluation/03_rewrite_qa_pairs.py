import argparse
import asyncio
import json
from pathlib import Path

import dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rewrite_question_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv(".oai.env")

QUESTION_REWRITE_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
QUESTION_REWRITE_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_REWRITE_SYS_TMPL,
        QUESTION_REWRITE_USER_TMPL
    ]
)


class QAPair(BaseModel):
    query: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The answer to the question.")


class State(TypedDict):
    document_id: str
    original_query: str
    original_answer: str
    query: str
    answer: str
    context: str


llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(QAPair)


def generate(state: State):
    prompt = prompt_template.invoke(
        {
            "context_str": state["context"],
            "question": state["original_query"],
            "answer": state["original_answer"]
        }
    )
    qa_pair = llm.invoke(prompt)
    return {
        "query": qa_pair.query,
        "answer": qa_pair.answer,
    }


async def main(input_path: str, output_dir: str, max_concurrency: int = 8):
    graph_builder = StateGraph(State)
    graph_builder.add_node(generate)
    graph_builder.add_edge(START, "generate")
    graph = graph_builder.compile()

    with open(input_path, 'r') as f:
        data = json.load(f)

    inputs = []
    qa_pair_dict = {}
    logger.info(f"{len(data)} documents loaded")
    for doc in data:
        if not doc["metadatas"]["verdict"]:
            continue
        original_query = doc["query"]
        original_answer = doc["ground_truth"]["content"]
        doc_id = doc["metadatas"]["document_id"]
        inputs.append({
            "document_id": doc_id,
            "original_query": original_query,
            "original_answer": original_answer,
            "context": doc['ground_truth']['contexts'][0],
        })
        qa_pair_dict[doc_id] = doc
    logger.info(f"{len(qa_pair_dict)} documents processed")

    outputs = await graph.abatch(inputs, config=RunnableConfig(max_concurrency=max_concurrency))
    logger.info(f"{len(outputs)} documents generated")
    qa_pairs = []
    for result in outputs:
        doc_id = result["document_id"]
        doc = qa_pair_dict[doc_id]
        doc["query"] = result["query"]
        doc['ground_truth']['content'] = result["answer"]
        doc["metadatas"]["original_query"] = result["original_query"]
        doc["metadatas"]["original_answer"] = result["original_answer"]
        qa_pairs.append(doc)

    # output_path = "data_eval/qa_pairs.v20241219.rewrite.json"
    output_path = Path(output_dir) / "qa_pairs.rewritten.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    logger.info(f"{len(qa_pairs)} documents rewritten to {output_path}")


if __name__ == '__main__':
    # python 03_rewrite_qa_pairs.py --input_path data_eval/v20250501/qa_pairs.validate.json --output_dir data_eval/v20250501
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(
        main(input_path=args.input_path, output_dir=args.output_dir, max_concurrency=args.max_concurrency)
    )
