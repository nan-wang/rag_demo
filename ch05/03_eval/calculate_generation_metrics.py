import json

import click
from tqdm import tqdm
from pathlib import Path
import dotenv
import re

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from keypoints_verify_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.output_parsers import StrOutputParser
from langchain import QAWithSourcesChain

from datamodels import KeyPoint


dotenv.load_dotenv()


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
    '--loyalty/--no-loyalty',
    default=False
)
@click.option(
    '--hallucination/--no-hallucination',
    default=False
)
@click.option(
    '--noise-sensitivity/--no-noise-sensitivity',
    default=False
)
@click.option(
    '--context-utility-ratio/--no-context-utility-ratio',
    default=False
)
@click.argument(
    'input_fn',
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
def main(num_docs, output_path, loyalty, hallucination, noise_sensitivity, context_utility_ratio, input_fn):
    with open(input_fn) as f:
        docs = json.load(f)
        response_loyalty_kp = []
        response_hallucination_kp = []
        response_noise_sensitivity_kp = []
        response_context_utility_ratio_kp = []
        for doc in docs[:num_docs]:
            question = doc["query"]
            answer = doc["ground_truth"]["content"]
            response = doc["response"]["content"]
            context = doc["response"]["contexts"][0]
            for k in doc["response"]["keypoints"]:
                response_loyalty_kp.append(
                    KeyPoint(question=question, answer=context, keypoint=k))
                response_hallucination_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )
                response_noise_sensitivity_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )
            for k in doc["ground_truth"]["keypoints"]:
                response_context_utility_ratio_kp.append(
                    (
                        KeyPoint(
                            question=question, answer=context, keypoint=k),
                        KeyPoint(
                            question=question, answer=answer, keypoint=k)
                    )
                )

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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    match = re.compile(r'\[\[\[([^\]]+)\]\]\]')

    chain = (prompt | llm | StrOutputParser())

    if loyalty:
        response_loyalty_list = []
        for kp in tqdm(response_loyalty_kp):
            result = chain.invoke({
                "question": kp.question,
                "answer": kp.answer,
                "keypoint": kp.keypoint
            })
            rsp = match.search(result)
            if rsp:
                kp.label = rsp.group(1)
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
            response_loyalty_list.append(kp)

        output_fn = Path(output_path) / "metrics" / "generation_loyalty.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([kp.dict() for kp in response_loyalty_list], f, indent=4, ensure_ascii=False)
        supported_kp = sum([1 for kp in response_loyalty_list if kp.label == "Relevant"])
        response_loyalty = supported_kp/len(response_loyalty_list)
        print(f"response_loyalty ↑: {response_loyalty:.3f}")

    if hallucination:
        result_list = []
        for kp_group in tqdm(response_hallucination_kp):
            # as long as one of the kp in the group is supported, the group is supported
            label = True
            for kp in tqdm(kp_group, leave=False):
                result = chain.invoke({
                    "question": kp.question,
                    "answer": kp.answer,
                    "keypoint": kp.keypoint
                })
                rsp = match.search(result)
                if rsp:
                    kp.label = rsp.group(1)
                    if kp.label == "Relevant":
                        label = False
                else:
                    print(f"Failed to extract the label for the keypoint: {result}")
            result_list.append((kp_group, label))

        output_fn = Path(output_path) / "metrics" / "generation_hallucination.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([{"details": [kp.dict() for kp in kp_g], "is_hallucination": l} for kp_g, l in result_list], f, indent=4, ensure_ascii=False)
        hallucination_kp = sum([label for kp_group, label in result_list])
        hallucination_score = hallucination_kp/len(result_list)
        print(f"hallucination score ↓: {hallucination_score:.3f}")

    if noise_sensitivity:
        result_list = []
        for claim_context, claim_ans in tqdm(response_noise_sensitivity_kp):
            label = False
            # check if the keypoint is supported by the context
            result = chain.invoke({
                "question": claim_context.question,
                "answer": claim_context.answer,
                "keypoint": claim_context.keypoint
            })
            rsp = match.search(result)
            if rsp:
                claim_context.label = rsp.group(1)
                if claim_context.label == "Relevant":
                    label = True
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
            # check if the keypoint is supported by the answer
            result = chain.invoke({
                "question": claim_ans.question,
                "answer": claim_ans.answer,
                "keypoint": claim_ans.keypoint
            })
            rsp = match.search(result)
            if rsp:
                claim_ans.label = rsp.group(1)
                if claim_ans.label == "Relevant":
                    label = False
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
            result_list.append(((claim_context, claim_ans), label))

        output_fn = Path(output_path) / "metrics" / "generation_noise_sensitivity.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([
                {
                    "details": [kp.dict() for kp in kp_g],
                    "is_noise": l
                } for kp_g, l in result_list
            ], f, indent=4, ensure_ascii=False)
        noise_sensitivity_kp = sum([label for kp_group, label in result_list])
        noise_sensitivity_score = noise_sensitivity_kp/len(result_list)
        print(f"noise sensitivity score ↓: {noise_sensitivity_score:.3f} ({noise_sensitivity_kp}/{len(result_list)})")

    if context_utility_ratio:
        result_list = []
        for claim_context, claim_ans in tqdm(response_context_utility_ratio_kp):
            # label used for calculating
            supported_by_cxt = True
            supported_by_cxt_and_ans = False
            # check if the keypoint is supported by the context
            result = chain.invoke({
                "question": claim_context.question,
                "answer": claim_context.answer,
                "keypoint": claim_context.keypoint
            })
            rsp = match.search(result)
            if rsp:
                claim_context.label = rsp.group(1)
                if claim_context.label != "Relevant":
                    supported_by_cxt = False
                elif claim_context.label == "Relevant":
                    supported_by_cxt_and_ans = True
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
            # check if the keypoint is supported by the answer
            result = chain.invoke({
                "question": claim_ans.question,
                "answer": claim_ans.answer,
                "keypoint": claim_ans.keypoint
            })
            rsp = match.search(result)
            if rsp:
                claim_ans.label = rsp.group(1)
                if claim_ans.label != "Relevant":
                    supported_by_cxt_and_ans = False
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
            result_list.append(((claim_context, claim_ans), supported_by_cxt, supported_by_cxt_and_ans))

        output_fn = Path(output_path) / "metrics" / "generation_context_utility_ratio.json"
        Path(output_fn).parent.mkdir(parents=True, exist_ok=True)
        with open(output_fn, 'w') as f:
            json.dump([{
                "claim_cxt": kp_cxt.dict(),
                "claim_ans": kp_ans.dict(),
                "supported_by_cxt": l_cxt,
                "supported_by_cxt_and_ans": l_ans
            } for (kp_cxt, kp_ans), l_cxt, l_ans in result_list], f, indent=4, ensure_ascii=False)
        context_utility_ratio_den = sum([l_cxt for (_, _), l_cxt, _ in result_list])
        context_utility_ratio_num = sum([l_ans for (_, _), _, l_ans in result_list])
        context_utility_ratio_score = context_utility_ratio_num / context_utility_ratio_den if context_utility_ratio_den else 0
        print(f"context utility ratio score ↑: {context_utility_ratio_score:.3f} ({context_utility_ratio_num}/{context_utility_ratio_den})")


if __name__ == '__main__':
    # python calculate_generation_metrics.py -n 100 -o data_metrics/v20241219/ch0503_naive --loyalty --hallucination --noise-sensitivity --context-utility-ratio data_metrics/v20241219/ch0503_naive/keypoints.json
    main()