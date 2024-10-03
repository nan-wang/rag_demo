from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_documents, get_chunks, format_docs

dotenv.load_dotenv()

vector_db_dir = '../data_chroma'
collection_name = 'test_db'


if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir,
        collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "巴黎奥运会的吉祥物有什么含义?",
        "output": "介绍巴黎奥运会的吉祥物",
    },
    {
        "input": "2004年奥运会和雅典共同竞争举办权的有哪些国家?",
        "output": "介绍2024年夏季奥运会的申办过程",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
template = """You are an expert at Olympic Games.\n
Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.\n
The generic step-back question should be answerable given a collection of wikipedia pages that contains \n
the information about the olympic games from 1980 to 2024.\n
Here are a few examples:
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            template,
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
generate_queries_step_back = prompt | llm | StrOutputParser()
query = "北京申办过几次奥运会?"

# 这种方法适合处理过于具体的query，但是document中只有宽泛描述的情况。
# step_back_query = "介绍北京申办奥运会的历史"
# "北京曾经于1993年申请举办过2000年夏季奥林匹克运动会，但最终在最后一轮的投票中以2票之差败于澳大利亚悉尼，此次申奥失败被称为“兵败蒙特卡洛”。"

response_prompt_template = """You are an expert of Olympic Games. \n
I am going to ask you a question. Your response should be comprehensive and not contradicted with \n
the following context if they are relevant. \n
Otherwise, ignore them if they are not relevant.

# Context:
{normal_context}

{step_back_context}

# Original Question: {question}
# Answer:"""
prompt = ChatPromptTemplate.from_template(response_prompt_template)


rag_chain = (
        {
            "normal_context": retriever | format_docs,
            "step_back_context": generate_queries_step_back | retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RunnableParallel(
    normal_contexts=RunnablePick("normal_context"),
    step_back_contexts=RunnablePick("step_back_context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

result = rag_chain.invoke(query)
print(result["answer"])

# 北京申办过三次奥运会。北京首次申办奥运会是在1993年，申办2000年夏季奥运会，但最终输给了悉尼。第二次申办是在2001年，成功获得2008年夏季奥运会的举办权。第三次申办是在2015年，成功获得2022年冬季奥运会的举办权，使北京成为首个既举办过夏季奥运会又举办过冬季奥运会的城市。