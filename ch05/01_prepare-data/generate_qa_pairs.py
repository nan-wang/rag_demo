import glob

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_all_splits():
    docs = []
    for file in glob.glob("data/*.txt"):
        loader = TextLoader(file)
        _docs = loader.load()
        docs += _docs

    print(f"Loaded {len(docs)} documents")

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


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
    "generate the relevant questions. "
)

result = QUESTION_GEN_USER_TMPL.format(
    context_str="This is a text snippet from wikipedia.")

prompt = ChatPromptTemplate.from_messages(
    messages=[
        QUESTION_GEN_SYS_TMPL,
        QUESTION_GEN_USER_TMPL
    ]
)

import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-2024-08-06")

# snippet_list = get_all_splits()
from langchain_chroma import Chroma

vectorstore = Chroma(persist_directory='data_chroma', collection_name='test_db')

ids = vectorstore.get()['ids']
import random
random.shuffle(ids)
selected_docs = {k: v for k, v in vectorstore.get(ids=ids[:3]).items() if k in ("ids", "metadatas", "documents")}

# `selected_docs` is A dict with the keys `"ids"`, `"embeddings"`, `"metadatas"`, `"documents"`. Convert a dictionary `selected_docs` to a list of dictionaries with the same keys.
selected_docs = [dict(zip(selected_docs, t)) for t in zip(*selected_docs.values())]

results = []
for doc in selected_docs:
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({
        "context_str": doc['documents'],
        "num_questions_per_chunk": 1
    })

    doc["question"] = result
    results.append(doc)

import json
# write results into a json file `qa_pairs.json`
with open("qa_pairs.json", "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# chain = (
#         prompt
#         | llm
#         | StrOutputParser()
# )
#
# results = chain.invoke({
#     "context_str": selected_doc['documents'][0],
#     "num_questions_per_chunk": 1
# })
#
# selected_doc["question"] = results
# print(selected_doc)