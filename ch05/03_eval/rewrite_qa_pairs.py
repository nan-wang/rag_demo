import dotenv
from rewrite_question_prompt import SYSTEM_PROMPT, USER_PROMPT
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pathlib import Path

dotenv.load_dotenv()

QUESTION_REWRITE_SYS_TMPL = (
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))
QUESTION_REWRITE_USER_TMPL = (
    HumanMessagePromptTemplate.from_template(USER_PROMPT))

prompt = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_REWRITE_SYS_TMPL,
        QUESTION_REWRITE_USER_TMPL
    ]
)

from langchain_core.pydantic_v1 import BaseModel, Field

class QAPair(BaseModel):
    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The answer to the question.")


llm = ChatOpenAI(model="gpt-4o-2024-08-06").with_structured_output(QAPair)

chain = (
        prompt
        | llm
)

input_path = "data_eval/qa_pairs.v20241219.json"
# load the json file at input_path
import json
with open(input_path, 'r') as f:
    data = json.load(f)

results = []
from tqdm import tqdm
for doc in tqdm(data):
    original_question = doc['query']
    original_answer = doc['ground_truth']['content']
    result = chain.invoke({
        "context_str": doc['ground_truth']['contexts'][0],
        "question": original_question,
        "answer": original_answer
    })
    qa_doc = doc
    # print(f"Original question: {doc['query']}")
    # print(f"Rewrite question: {result.question}")
    # print(f"Original answer: {doc['ground_truth']['content']}")
    # print(f"Rewrite answer: {result.answer}")
    qa_doc['query'] = result.question
    qa_doc['ground_truth']['content'] = result.answer
    qa_doc['metadatas']['original_question'] = original_question
    qa_doc['metadatas']['original_answer'] = original_answer
    results.append(qa_doc)

output_path = "data_eval/qa_pairs.v20241219.rewrite.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

