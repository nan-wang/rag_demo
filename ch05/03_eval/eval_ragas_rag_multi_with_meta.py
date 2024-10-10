import json
from pathlib import Path

import dotenv
from datasets import Dataset
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_precision,
)

from utils import load_documents, get_chunks, format_docs, split_contexts

dotenv.load_dotenv()

# load the test data
examples = []
with open("data_eval/qa_pairs.v20241009.json", "r") as f:
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

# build the rag chain
vector_db_dir = '../data_chroma_multi_with_metadata'
collection_name = 'test_db'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir,
        collection_name=collection_name)

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year of the Olympic Games the document is about.",
        type="int"),
    AttributeInfo(
        name="season",
        description="The season of the Olympic Games the document is about.",
        type="str"),
]

document_content_description = \
    "General information about the Olympic Games between 1980 and 2024 from Wikipedia in Chinese."

llm = ChatOpenAI(model="gpt-4o-2024-08-06")

retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    search_kwargs={"k": 10},
)

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableParallel(
    contexts=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

question = []
answer = []
ground_truth = []
contexts = []
for example in examples:
    q = example["query"]
    result = rag_chain.invoke(q)
    question.append(q)
    answer.append(result["answer"])
    ground_truth.append(example["ground_truth"])
    contexts.append(split_contexts(result["contexts"]))

data = {
    "question": question,
    "answer": answer,
    "ground_truth": ground_truth,
    "contexts": contexts,
}

dataset = Dataset.from_dict(data)

answer_relevancy.llm = None
context_recall.llm = None
context_precision.llm = None
answer_relevancy.embeddings = None
context_recall.embeddings = None
context_precision.embeddings = None

# evaluate the results
result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        answer_relevancy,
    ],
    llm=ChatOpenAI(model="gpt-4o-2024-08-06"),
    embeddings=OpenAIEmbeddings(),
)

eval_df = result.to_pandas()

eval_df.to_json("eval_results.multi-meta.v20241010.json", orient="records", indent=4, force_ascii=False)

mask = ((eval_df["answer_relevancy"] < 0.5) | (eval_df["context_precision"] < 0.1))
wrong_answers = eval_df[mask]

wrong_answers.to_json("wrong_answers.multi-meta.v20241010.json", orient="records", indent=4, force_ascii=False)
