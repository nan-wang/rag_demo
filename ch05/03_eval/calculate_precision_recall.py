import json
from tqdm import tqdm
from pathlib import Path
import dotenv

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
    rsp_kp = []
    ans_kp = []
    context_precision_kp = []
    context_recall_kp = []
    for doc in docs[:2]:
        question = doc["query"]
        answer = doc["ground_truth"]["content"]
        response = doc["response"]["content"]
        context = doc["response"]["contexts"][0]
        for k in doc["response"]["keypoints"]:
            rsp_kp.append(
                KeyPoint(question=question, answer=answer, keypoint=k))
        for k in doc["ground_truth"]["keypoints"]:
            ans_kp.append(
                KeyPoint(question=question, answer=response, keypoint=k))
            context_recall_kp.append(
                KeyPoint(question=question, answer=context, keypoint=k))
        for ctx in context.split("\n"):
            ctx = ctx.strip("\n")
            if not ctx:
                continue
            cur_context_kp = []
            for k in doc["response"]["keypoints"]:
                cur_context_kp.append(
                    KeyPoint(question=question, answer=ctx, keypoint=k))
            context_precision_kp.append(cur_context_kp)

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

llm = ChatOpenAI(model="gpt-4o-2024-08-06")
import re
match = re.compile(r'\[\[\[([^\]]+)\]\]\]')

chain = (prompt | llm | StrOutputParser())

cal_precision = False
cal_recall = False
cal_context_precision = False
cal_context_recall = True

# calculate the precision
if cal_precision:
    precision_list = []

    for kp in tqdm(rsp_kp):
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
        precision_list.append(kp)

    output_path = "data_eval/results.naive_rag.v20241219.keypoints.precision.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([kp.dict() for kp in precision_list], f, indent=4, ensure_ascii=False)
    supported_kp = sum([1 for kp in precision_list if kp.label == "Relevant"])
    precision = supported_kp/len(precision_list)
    print(f"precision: {precision}")

if cal_recall:
    recall_list = []

    for kp in tqdm(ans_kp):
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
        recall_list.append(kp)

    output_path = "data_eval/results.naive_rag.v20241219.keypoints.recall.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([kp.dict() for kp in recall_list], f, indent=4, ensure_ascii=False)
    supported_kp = sum([1 for kp in recall_list if kp.label == "Relevant"])
    recall = supported_kp/len(recall_list)
    print(f"recall: {recall}")

if cal_precision and cal_recall:
    f1 = 2 * precision * recall / (precision + recall)
    print(f"f1: {f1}")

if cal_context_precision:
    context_precision_list = []
    for kp_group in tqdm(context_precision_kp):
        # as long as one of the kp in the group is supported, the group is supported
        label = False
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
                    label = True
            else:
                print(f"Failed to extract the label for the keypoint: {result}")
        context_precision_list.append((kp_group, label))

    output_path = "data_eval/results.naive_rag.v20241219.keypoints.context_precision.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([([kp.dict() for kp in kp_g], l) for kp_g, l in context_precision_list], f, indent=4, ensure_ascii=False)
    supported_kp = sum([label for kp_group, label in context_precision_list])
    context_precision = supported_kp/len(context_precision_list)
    print(f"context_precision: {context_precision}")

if cal_context_recall:
    context_recall_list = []
    for kp in tqdm(context_recall_kp):
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
        context_recall_list.append(kp)

    output_path = "data_eval/results.naive_rag.v20241219.keypoints.context_recall.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([kp.dict() for kp in context_recall_list], f, indent=4, ensure_ascii=False)
    supported_kp = sum([1 for kp in context_recall_list if kp.label == "Relevant"])
    context_recall = supported_kp/len(context_recall_list)
    print(f"context_recall: {context_recall}")