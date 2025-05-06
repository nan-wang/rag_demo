import click
import dotenv
import tqdm
import json
from loguru import logger

from keypoints_extract_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from datamodels import KeyPoints


dotenv.load_dotenv(".oai.env")


@click.command()
@click.option(
    '--num_docs',
    '-n',
    default=-1,
    help='The number of documents to be processed.')
@click.option(
    '--output_path',
    '-o',
    default="./metrics",
    help='The output file path.',
    type=click.Path(file_okay=False, dir_okay=True, writable=True)
)
@click.option(
    '--ground-truth/--no-ground-truth',
    default=False,
    help='Whether to extract keypoints from ground truth or response.'
)
@click.option(
    '--response/--no-response',
    default=False,
    help='Whether to extract keypoints from ground truth or response.'
)
@click.argument(
    'input_fn',
    default="response.json"
)
def main(num_docs, output_path, ground_truth, response, input_fn):
    KE_SYS_TMPL = (
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

    KE_USER_TMPL = (
        HumanMessagePromptTemplate.from_template(USER_PROMPT))

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            KE_SYS_TMPL,
            KE_USER_TMPL
        ]
    )

    llm = ChatOpenAI(model="gpt-4.1-mini").with_structured_output(KeyPoints)

    chain = (prompt | llm)

    with open(input_fn, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded from {input_fn}")

    results = []
    logger.info(f"Selected {num_docs if num_docs!=-1 else len(data)} from {len(data)} documents")
    for doc in tqdm.tqdm(data[:num_docs]):
        question = doc['query']
        if ground_truth:
            answer = doc['ground_truth']['content']
            result = chain.invoke({
                "question": question,
                "answer": answer
            })
            try:
                doc["ground_truth"]["keypoints"] = result.keypoints
            except Exception as e:
                logger.info(f"Error: {e}")
                logger.info(f"Failed to extract keypoints from ground truth for result: {result}")
                continue
            doc["ground_truth"]["keypoints"] = result.keypoints

        if response:
            response = doc['response']['content']
            result = chain.invoke({
                "question": question,
                "answer": response
            })
            doc["response"]["keypoints"] = result.keypoints
        results.append(doc)

    output_fn = Path(output_path) / "keypoints.json"
    Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
    with open(output_fn, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved the results to {output_fn}")


if __name__ == "__main__":
    # python 01_extract_keypoints.py -n 10 -g -r -o data_metrics/v20241219/toy data_metrics/v20241219/ch0503_naive/response.json
    main()