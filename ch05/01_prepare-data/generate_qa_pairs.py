import json
import random
from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

QUESTION_GEN_SYS_TMPL = SystemMessagePromptTemplate.from_template("""\
You are a Teacher. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.\
ALL THE QUESTIONS MUST BE IN CHINESE.\
""")

QUESTION_GEN_USER_TMPL = HumanMessagePromptTemplate.from_template(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate the relevant question and the answer. \n"
    "The question and answer should be in Chinese. \n"
    "Return the results in JSON format. \n"
    "The JSON object must contain the following keys: \n"
    "- 'question': a string, the question generated from the context. \n"
    "- 'answer': a string, the answer to the question. \n"
)

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

vectorstore = Chroma(persist_directory='data_chroma', collection_name='test_db')

ids = vectorstore.get()['ids']

random.shuffle(ids)
selected_docs = {k: v for k, v in vectorstore.get(ids=ids[:20]).items() if k in ("ids", "metadatas", "documents")}

# `selected_docs` is A dict with the keys `"ids"`, `"embeddings"`, `"metadatas"`, `"documents"`. Convert a dictionary `selected_docs` to a list of dictionaries with the same keys.
selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

results = []
for doc in selected_docs:
    chain = (
            prompt
            | llm
    )

    result = chain.invoke({
        "context_str": doc['documents'],
        "num_questions_per_chunk": 1
    })

    doc["question"] = result.question
    doc["answer"] = result.answer
    results.append(doc)

output_path = "data_eval/qa_pairs.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
