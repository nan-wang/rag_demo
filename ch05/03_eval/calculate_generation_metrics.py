import json
from tqdm import tqdm
from pathlib import Path
import dotenv
import re

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


# load the data from the file, data_eval/qa_pairs.v20241009.keypoints.json
with open("data_eval/results.naive_rag.v20241219.keypoints.json", "r") as f:
    docs = json.load(f)
    response_loyalty_kp = []
    response_hallucination_kp = []
    response_noise_sensitivity_kp = []
    for doc in docs[:2]:
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

cal_loyalty = False
cal_hallucination = True
cal_noise_sensitivity = False

if cal_loyalty:
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

    output_path = "data_eval/results.naive_rag.v20241219.metrics.generation_loyalty.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([kp.dict() for kp in response_loyalty_list], f, indent=4, ensure_ascii=False)
    supported_kp = sum([1 for kp in response_loyalty_list if kp.label == "Relevant"])
    response_loyalty = supported_kp/len(response_loyalty_list)
    print(f"response_loyalty: {response_loyalty}")

if cal_hallucination:
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

    output_path = "data_eval/results.naive_rag.v20241219.metrics.hallucination.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([([kp.dict() for kp in kp_g], l) for kp_g, l in result_list], f, indent=4, ensure_ascii=False)
    hallucination_kp = sum([label for kp_group, label in result_list])
    hallucination_score = hallucination_kp/len(result_list)
    print(f"hallucination score: {hallucination_score}")

if cal_noise_sensitivity:
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

    output_path = "data_eval/results.naive_rag.v20241219.metrics.noise_sensitivity.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([([kp.dict() for kp in kp_g], l) for kp_g, l in result_list], f, indent=4, ensure_ascii=False)
    noise_sensitivity_kp = sum([label for kp_group, label in result_list])
    noise_sensitivity_score = noise_sensitivity_kp/len(result_list)
    print(f"noise sensitivity score: {noise_sensitivity_score}")