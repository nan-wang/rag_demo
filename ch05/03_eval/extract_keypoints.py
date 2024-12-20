import dotenv
import tqdm
import json

from keypoints_extract_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain_core.pydantic_v1 import BaseModel, Field


dotenv.load_dotenv()

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

class KeyPoints(BaseModel):
    keypoints: list = Field(..., description="The keypoints extracted from the context.")

llm = ChatOpenAI(model="gpt-4o-2024-08-06").with_structured_output(KeyPoints)

chain = (prompt | llm)

input_path = "data_eval/results.v20241219.naive_rag.json"
with open(input_path, 'r') as f:
    data = json.load(f)

results = []
for doc in tqdm.tqdm(data):
    question = doc['query']
    answer = doc['ground_truth']['content']
    result = chain.invoke({
        "question": question,
        "answer": answer
    })
    doc["ground_truth"]["keypoints"] = result.keypoints

    response = doc['response']['content']
    result = chain.invoke({
        "question": question,
        "answer": response
    })
    doc["response"]["keypoints"] = result.keypoints
    results.append(doc)

output_path = "data_eval/results.naive_rag.v20241219.keypoints.json"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)