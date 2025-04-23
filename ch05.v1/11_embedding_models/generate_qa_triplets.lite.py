import json
from itertools import cycle
import random

import dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

from langchain_core.pydantic_v1 import BaseModel, Field

SYSTEM_PROMPT = """
你是一个国际奥林匹克委员会的专家。
你的任务是为即将到来的知识竞赛设计问题和答案。
问题应该涵盖给定参考文档中的内容。
问题和答案都不应当超出参考文档的内容。
"""

USER_PROMPT = """
在下面的文档中，您将看到一些上下文信息。
---------------------
{context_str}
---------------------
给定上下文信息，但不考虑先验知识，生成相关问题、答案以及错误上下文信息。
- 问题和答案必须是中文的。
- 问题长度至少{length}个字。
- 问题难度是{clarity}级别的。
- 问题必须是没有歧义的。
- 题目是针对{difficulty}教育背景的学生设计的。
- 问题和答案必须是基于上下文信息的。
- 错误上下文信息必须是与问题相关的，但是不能回答问题。
- 错误上下文信息至少包含64个字符，但不超过512个字符。
- 错误上下文信息必须包含与正确上下文类似的开头格式，直到"content: "。
其中,"article_title"可以替换为其他年份的夏季或冬季奥运会。
以JSON格式返回结果。
JSON对象必须包含以下键：
- 'question'：一个字符串，从上下文生成的问题。
- 'answer'：一个字符串，问题的答案。
- 'negative_document'：一个字符串，错误上下文信息。
无法回答问题，但是与上下文信息相似的文档。 
错误上下文信息必须包含与正确上下文类似的开头格式，直到"content: "!!!

Your output must always be a JSON object only, 
do not explain yourself or output anything else. 
Be creative!
"""

input_fn = "data_chroma"  # the index file path
num_docs = 20000  # the number of documents to be generated


class QATriplet(BaseModel):
    question: str = Field(..., description="The question generated from the context.")
    answer: str = Field(..., description="The correct answer to the question.")
    negative_document: str = Field(
        ..., description="The wrong context not related to the question."
    )


QUESTION_GEN_SYS_TMPL = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
QUESTION_GEN_USER_TMPL = HumanMessagePromptTemplate.from_template(USER_PROMPT)
prompt = ChatPromptTemplate.from_messages(
    messages=[QUESTION_GEN_SYS_TMPL, QUESTION_GEN_USER_TMPL]
)
llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(QATriplet)
vectorstore = Chroma(persist_directory=input_fn, collection_name="olympic_games")
ids = vectorstore.get()["ids"]
random.shuffle(ids)
selected_docs = {
    k: v
    for k, v in vectorstore.get(ids=ids).items()
    if k in ("ids", "metadatas", "documents")
}
selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]
results_triplet = []
# random select 10000 docs from selected_docs with repetition
for doc in cycle(selected_docs):
    if len(results_triplet) >= num_docs:
        break
    length = random.choice([8, 16, 32])
    clarity = random.choice(["简单", "基础", "困难"])
    difficulty = random.choice(["小学", "初中", "高中", "大学", "研究生博士"])
    chain = prompt | llm
    try:
        result = chain.invoke(
            {
                "context_str": doc["documents"],
                "length": length,
                "clarity": clarity,
                "difficulty": difficulty,
            }
        )
        qa_triplet = {
            "anchor": result.question,
            "positive": doc["documents"],
            "negative": result.negative_document,
        }
        results_triplet.append(qa_triplet)
    except Exception as e:
        print(e)
        continue

with open("qa_triplets.json", "w") as f:
    json.dump(results_triplet, f, indent=4, ensure_ascii=False)
