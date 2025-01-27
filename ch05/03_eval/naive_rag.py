import dotenv
import json
from pathlib import Path
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_core.output_parsers import StrOutputParser

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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

retrieved_docs = retriever.invoke("奥运会金牌的挂带有哪些设计?")

llm = ChatOpenAI(model="gpt-4o-mini")
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

# keep the context and return
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough())
    | RunnableParallel(
        context=RunnablePick("context"),
        question=RunnablePick("question"),
        answer=prompt | llm | StrOutputParser())
)

results = []
# open the json file at data_eval/qa_pairs.v20241219.rewrite.json
with open("data_eval/v20241219/qa_pairs_rewrite.json", "r") as f:
    qa_pairs = json.load(f)
    for doc in tqdm(qa_pairs):
        query = doc["query"]
        result = rag_chain.invoke(query)
        doc["response"] = {
            "content": result["answer"],
            "contexts": [result["context"],]
        }
        results.append(doc)

output_path = "data_metrics/v20241219/ch0503_naive/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
