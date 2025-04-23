import json
import random
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from synthetic_data_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv()

from langchain_core.pydantic_v1 import BaseModel, Field

class QATriplet(BaseModel):
    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The correct answer to the question.")
    negative_document: str = Field(..., description="The wrong context not related to the question.")

QUESTION_GEN_SYS_TMPL = (
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

QUESTION_GEN_USER_TMPL = (
    HumanMessagePromptTemplate.from_template(USER_PROMPT))


import click

@click.command()
@click.option(
    '--num_docs',
    '-n',
    default=-1,
    help='The number of documents to be generated.'
)
@click.option(
    '--output_path',
    '-o',
    default="./data_finetuning",
    help='The output file path.',
    type=click.Path(file_okay=False, dir_okay=True, writable=True)
)
@click.argument(
    'input_fn',
    default="../data_chroma_add_meta_info_128"
)
def main(num_docs: int, output_path: str, input_fn: str):
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            QUESTION_GEN_SYS_TMPL,
            QUESTION_GEN_USER_TMPL
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(QATriplet)

    vectorstore = Chroma(persist_directory=input_fn, collection_name='olympic_games')

    ids = vectorstore.get()['ids']

    print(f"Total number of documents: {len(ids)}")

    random.shuffle(ids)
    selected_docs = {k: v for k, v in vectorstore.get(ids=ids).items() if k in ("ids", "metadatas", "documents")}

    selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

    results = []
    results_triplet = []

    import tqdm
    # random select 10000 docs from selected_docs with repetition
    from itertools import cycle
    for doc in tqdm.tqdm(cycle(selected_docs)):
        if len(results) >= num_docs:
            break
        length = random.choice([8, 16, 32])
        clarity = random.choice(["简单", "基础", "困难"])
        difficulty = random.choice(["小学", "初中", "高中", "大学", "研究生博士"])

        chain = (
                prompt
                | llm
        )

        try:
            result = chain.invoke({
                "context_str": doc['documents'],
                "length": length,
                "clarity": clarity,
                "difficulty": difficulty
            })

            qa_doc = {
                "query": result.question,
                "ground_truth": {
                    "contexts": [doc["documents"],],
                    "content": result.answer
                },
                "metadatas": {
                    "length": length,
                    "clarity": clarity,
                    "difficulty": difficulty,
                    "document_id": doc["ids"]
                }
            }
            qa_triplet = {
                "anchor": result.question,
                "positive": doc["documents"],
                "negative": result.negative_document
            }
            results.append(qa_doc)
            results_triplet.append(qa_triplet)
        except Exception as e:
            print(e)
            continue

    output_path = Path(output_path)
    output_fn = output_path / "qa_pairs.json"
    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fn, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    output_fn = output_path / "qa_triplets.json"
    with open(output_fn, "w") as f:
        json.dump(results_triplet, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    # python generate_qa_triplets.py -n 3 -o ./data_finetuning/v20250202.v1 ../data_chroma_add_meta_info_128
    main()