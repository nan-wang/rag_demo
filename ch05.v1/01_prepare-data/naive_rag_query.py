import dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

VECTOR_DB_DIR = "data_chroma"
COLLECTION_NAME = "olympic_games"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


vectorstore = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=OpenAIEmbeddings(),
    create_collection_if_not_exists=False,
    collection_name=COLLECTION_NAME,
)
print(f"Loaded {vectorstore._chroma_collection.count()} documents")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 300})

retrieved_docs = retriever.invoke("奥运会金牌的挂带有哪些设计?")

print(f"Retrieved {len(retrieved_docs)} documents")
for idx, doc in enumerate(retrieved_docs):
    print(f"#{idx}: {repr(doc.page_content)}")
exit(0)

llm = ChatOpenAI(model="gpt-4o-2024-08-06")
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

rag_chain = (
        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        | prompt
        | llm
        | StrOutputParser()
)

query = "2024年巴黎奥运会的开幕式是哪一天?"
result = rag_chain.invoke(query)
print(result)


# rag_chain.get_graph().print_ascii()
# rag_chain.get_graph().draw_mermaid_png(output_file_path="rag_chain.png")
# 输出结果
# Loaded 245 documents
# 2024年巴黎奥运会的开幕式将于2024年7月26日举行。
