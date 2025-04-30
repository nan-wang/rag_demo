import asyncio
import argparse
import json
from pathlib import Path
from loguru import logger

import dotenv
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from validate_question_answer_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv(".oai.env")

QUESTION_VALIDATE_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
QUESTION_VALIDATE_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)

prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_VALIDATE_SYS_TMPL,
        QUESTION_VALIDATE_USER_TMPL
    ]
)


class QAFeedback(BaseModel):
    feedback: str = Field(..., description="Feedback for the question.")
    verdict: int = Field(..., description="Score for the question.")


llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(QAFeedback)


class State(TypedDict):
    context: str
    query: str
    answer: str
    feedback: str
    response: str
    document_id: str
    verdict: int


def verify(state: State):
    prompt = prompt_template.invoke(
        {
            "context_str": state['context'],
            "question": state['query'],
            "answer": state['answer'],
        }
    )
    response = llm.invoke(prompt)
    return {
        "feedback": response.feedback,
        "verdict": response.verdict,
    }


async def main(input_path: str, output_path: str, max_concurrency: int = 8):
    graph_builder = StateGraph(State)
    graph_builder.add_node(verify)
    graph_builder.add_edge(START, "verify")
    graph_builder.add_edge("verify", END)
    graph = graph_builder.compile()

    with open(input_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Read {len(data)} questions from {input_path}")

    inputs = [
        {
            "query": doc['query'],
            "answer": doc['ground_truth']['content'],
            "context": doc['ground_truth']['contexts'][0],
            "document_id": doc['metadatas']['document_id']
        } for doc in data
    ]
    outputs = await graph.abatch(inputs, config=RunnableConfig(max_concurrency=max_concurrency))
    result_dict = {}
    for result in outputs:
        result_dict[result['document_id']] = result
    results = []
    for doc in data:
        document_id = doc['metadatas']['document_id']
        output_doc = doc
        output_doc['metadatas']['feedback'] = result_dict[document_id]['feedback']
        output_doc['metadatas']['verdict'] = result_dict[document_id]['verdict']
        results.append(output_doc)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Wrote {len(results)} questions to {output_path}")


if __name__ == '__main__':
    # python 03_validate_qa_pairs.py --input_path data_eval/v20250501/qa_pairs.json --output_path data_eval/v20250501/qa_pairs.validate.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.input_path, args.output_path, args.max_concurrency))
