import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

dotenv.load_dotenv()

vector_db_dir = '../data_chroma_add_meta_info'
collection_name = 'olympic_games'

vectorstore = Chroma(
    persist_directory=vector_db_dir,
    embedding_function=OpenAIEmbeddings(),
    create_collection_if_not_exists=False,
    collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


prompt_template_for_decomposition = """You are a helpful assistant that generates multiple sub-questions related to an input question.\n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
The sub-problems should be answerable by a collection of documents that contains the information about the olympic games from 1980 to 2024. \n
Generate multiple search queries related to: {question} \n
Output:"""

prompt_decomposition = ChatPromptTemplate.from_template(prompt_template_for_decomposition)
llm = ChatOpenAI(model="gpt-4o-mini")

generate_queries_decomposition = (
        prompt_decomposition
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
)


def format_docs(retrieval_docs_from_multi_query):
    docs = []
    for docs_per_query in retrieval_docs_from_multi_query:
        docs += docs_per_query
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx+1}]{doc_str}")
    return "\n\n".join(output_list)

query = "列举过去5届夏季奥运会吉祥物的名称?"
result = generate_queries_decomposition.invoke(query)
retrieval_chain = generate_queries_decomposition | retriever.map()

rag_chain = (
        RunnableParallel(
            context=retrieval_chain | format_docs,
            question=RunnablePassthrough())
        | RunnableParallel(
    contexts=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

query = "列举过去5届夏季奥运会吉祥物的名称?"
result = rag_chain.invoke(query)
print(result["answer"])

# 'question': '列举过去5届夏季奥运会吉祥物的名称?',
# 'answer':
# 过去5届夏季奥运会的吉祥物名称分别是：
# 1. 2008年：福娃（五个形象：贝贝、晶晶、欢欢、迎迎、妮妮）
# 2. 2012年：文洛克（Wenlock）和曼德维尔（Mandeville）
# 3. 2016年：费尼希斯（Vinicius）
# 4. 2020年：梅花鹿（Miraitowa）
# 5. 2024年：奥林匹克弗里热（Phryges）

