import json
import click

from tqdm import tqdm
from pathlib import Path
import dotenv
import re
from utils import dump_metrics

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from keypoints_verify_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.output_parsers import StrOutputParser


dotenv.load_dotenv()
class KeyPoint(BaseModel):
    question: str = Field(..., description="The question.")
    answer: str = Field(..., description="The answer.")
    keypoint: str = Field(..., description="The keypoint related to the question which should be covered by the answer")
    label: str = Field("Relevant", description="The label indicating whether the answer covers the keypoint.")


def verify_keypoints(keypoints, lc_chain):
    match = re.compile(r'\[\[\[([^\]]+)\]\]\]')
    results = []
    for kp in tqdm(keypoints):
        result = lc_chain.invoke({
            "question": kp.question,
            "answer": kp.answer,
            "keypoint": kp.keypoint
        })
        rsp = match.search(result)
        if rsp:
            kp.label = rsp.group(1)
        else:
            print(f"Failed to extract the label for the keypoint: {result}")
        results.append(kp)
    return results


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
    '--precision/--no-precision',
    default=True
)
@click.option(
    '--recall/--no-recall',
    default=True
)
@click.argument(
    'input_fn',
    default="keypoints.json")
def main(num_docs, output_path, precision, recall, input_fn):
    # load the data from the file, data_eval/qa_pairs.v20241009.keypoints.json
    with open(input_fn, "r") as f:
        docs = json.load(f)
        rsp_kp = []
        ans_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            answer = doc["ground_truth"]["content"]
            response = doc["response"]["content"]
            for k in doc["response"]["keypoints"]:
                rsp_kp.append(
                    KeyPoint(question=question, answer=answer, keypoint=k))
            for k in doc["ground_truth"]["keypoints"]:
                ans_kp.append(
                    KeyPoint(question=question, answer=response, keypoint=k))

    KV_SYS_TMPL = (
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

    KV_USER_TMPL = (
        HumanMessagePromptTemplate.from_template(USER_PROMPT))

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            KV_SYS_TMPL,
            KV_USER_TMPL
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    chain = (prompt | llm | StrOutputParser())

    # calculate the precision
    if precision:
        precision_list = verify_keypoints(rsp_kp, chain)
        dump_metrics(
            precision_list,
            Path(output_path) / "metrics" / "global_precision.json")
        supported_kp = sum([1 for kp in precision_list if kp.label == "Relevant"])
        precision_score = supported_kp / len(precision_list)
        print(f"precision: {precision_score:.3f}")

    if recall:
        recall_list = verify_keypoints(ans_kp, chain)
        dump_metrics(
            recall_list,
            Path(output_path) / "metrics" / "global_recall.json")
        supported_kp = sum([1 for kp in recall_list if kp.label == "Relevant"])
        recall_score = supported_kp / len(recall_list)
        print(f"recall: {recall_score:.3f}")

    if precision and recall:
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score)
        print(f"f1: {f1:.3f}")


if __name__ == '__main__':
    main()


