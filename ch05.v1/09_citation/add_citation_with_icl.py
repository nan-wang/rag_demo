import dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick

dotenv.load_dotenv()

VECTOR_DB_DIR = "../01_prepare-data/data_chroma"
COLLECTION_NAME = "olympic_games"


def format_docs(docs):
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"[doc_{idx+1}]{doc_str}")
    return "\n\n".join(output_list)


vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=OpenAIEmbeddings(),
    create_collection_if_not_exists=False,
    collection_name=COLLECTION_NAME,
)
print(f"Loaded {vectorstore._chroma_collection.count()} documents")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini")
xlm_template = """
You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, \
answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
that justify the answer. Use the following format for your final output:

<cited_answer>
    <answer></answer>
    <citations>
        <citation><source_id></source_id><quote></quote></citation>
        <citation><source_id></source_id><quote></quote></citation>
        ...
    </citations>
</cited_answer>

Question: {question}
Here are the Wikipedia articles: {context}
"""

xlm_prompt = ChatPromptTemplate.from_template(xlm_template)

from langchain_core.output_parsers import XMLOutputParser

def format_docs_xml(docs):
    output_list = []
    for idx, doc in enumerate(docs):
        doc_str = doc.page_content.replace("\n", " ")
        output_list.append(f"<doc_{idx+1}>{doc_str}")
    return "\n\n".join(output_list)

# keep the context and return
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs_xml,
        question=RunnablePassthrough())
    | RunnableParallel(
        context=RunnablePick("context"),
        question=RunnablePick("question"),
        answer=xlm_prompt | llm | XMLOutputParser())
)
query = "2024年巴黎奥运会的体育项目图标在设计上有哪些独特之处？"
result = rag_chain.invoke(query)
print(result)
