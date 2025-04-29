import json
import random
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from synthetic_data_prompt import SYSTEM_PROMPT, USER_PROMPT

dotenv.load_dotenv()

QUESTION_GEN_SYS_TMPL = (
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT))

QUESTION_GEN_USER_TMPL = (
    HumanMessagePromptTemplate.from_template(USER_PROMPT))

prompt = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_GEN_SYS_TMPL,
        QUESTION_GEN_USER_TMPL
    ]
)

from langchain_core.pydantic_v1 import BaseModel, Field

class QAPair(BaseModel):
    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The answer to the question.")

llm = ChatOpenAI(model="gpt-4o-2024-08-06").with_structured_output(QAPair)

vectorstore = Chroma(persist_directory='../data_chroma_multi', collection_name='test_db')

ids = vectorstore.get()['ids']

print(f"Total number of documents: {len(ids)}")

random.shuffle(ids)
selected_docs = {k: v for k, v in vectorstore.get(ids=ids[:300]).items() if k in ("ids", "metadatas", "documents")}

selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

results = []

import tqdm
for doc in tqdm.tqdm(selected_docs):
    length = random.choice([8, 16, 32])
    clarity = random.choice(["简单", "基础", "困难"])
    difficulty = random.choice(["小学", "初中", "高中", "大学", "研究生博士"])

    chain = (
            prompt
            | llm
    )

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
    results.append(qa_doc)

output_path = "data_eval/qa_pairs.v20241219.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
