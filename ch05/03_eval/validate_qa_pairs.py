import dotenv
from validate_question_answer import SYSTEM_PROMPT, USER_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pathlib import Path

dotenv.load_dotenv()

QUESTION_VALIDATE_SYS_TMPL = (
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))
QUESTION_VALIDATE_USER_TMPL = (
    HumanMessagePromptTemplate.from_template(USER_PROMPT))

prompt = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_VALIDATE_SYS_TMPL,
        QUESTION_VALIDATE_USER_TMPL
    ]
)

from langchain_core.pydantic_v1 import BaseModel, Field

class QAFeedback(BaseModel):
    feedback: str = Field(..., description="Feedback for the question.")
    verdict: int = Field(..., description="Score for the question.")

llm = ChatOpenAI(model="gpt-4o-2024-08-06").with_structured_output(QAFeedback)

chain = (prompt | llm)

input_path = "data_eval/qa_pairs.v20241219.json"

import json
with open(input_path, 'r') as f:
    data = json.load(f)

results = []
from tqdm import tqdm
for doc in tqdm(data):
    result = chain.invoke({
        "context_str": doc['ground_truth']['contexts'][0],
        "question": doc['query'],
        "answer": doc['ground_truth']['content']
    })
    qa_doc = doc
    qa_doc["metadatas"]['feedback'] = result.feedback
    qa_doc["metadatas"]['verdict'] = result.verdict
    results.append(qa_doc)

output_path = "data_eval/qa_pairs.v20241219.validate.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)