import sys
import json

import dotenv
from pathlib import Path
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from utils import load_documents, get_chunks, format_docs
dotenv.load_dotenv()

vector_db_dir = 'data_chroma'
collection_name = 'test_db'

from langchain_core.documents import Document



docs = load_documents("data/*.txt")
print(f"Loaded {len(docs)} documents")

chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")


if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(), create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir, collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# retrieved_docs = retriever.invoke("中国第一次参加冬奥会是哪一年?")
#
# print(f"Retrieved {len(retrieved_docs)} documents")
# for doc in retrieved_docs:
#     print(f"Retrieved doc, {repr(doc.page_content[:100])}")
#     print(f"Retrieved doc meta, {doc.metadata}")


llm = ChatOpenAI(model="gpt-4o-2024-08-06")
prompt = hub.pull("rlm/rag-prompt")


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
)

query = "中华人民共和国第一次参加冬奥会是哪一年?"

result = rag_chain.invoke(query)
print(result)

sys.exit(0)

# load the evaluation data from `data_eval/qa_pairs.json`
examples = []
with open("data_eval/qa_pairs.json", "r") as f:
    qa_pairs = json.load(f)
    for qa_pair in qa_pairs:
        query = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        source_documents = qa_pair["documents"]
        examples.append(
            {
                "query": query,
                "ground_truth": ground_truth,
                "source_documents": source_documents,
            }
        )

from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

faithfulness_chain = EvaluatorChain(metric=faithfulness)
answer_relevancy_chain = EvaluatorChain(metric=answer_relevancy)
# context_precision_chain = EvaluatorChain(metric=context_precision)
# context_recall_chain = EvaluatorChain(metric=context_recall)

result = rag_chain.invoke(examples[0]["query"])
print(result)
eval_result = faithfulness_chain.invoke({"question": examples[0]["query"], "answer": result, "context": examples[0]["source_documents"]})
print(eval_result)
