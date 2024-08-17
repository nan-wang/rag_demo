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
You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided.\
""")

QUESTION_GEN_USER_TMPL = HumanMessagePromptTemplate.from_template(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate the relevant questions. "
)

QA_PROMPT_TMPL = HumanMessagePromptTemplate.from_template(
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query}\n"
    "Answer: "
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

snippet_list = get_all_splits()
chain = (
        prompt
        | llm
        | StrOutputParser()
)
chain.invoke({
    "context_str": snippet_list[0].page_content,
    "num_questions_per_chunk": 3
})
