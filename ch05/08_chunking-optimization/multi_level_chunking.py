from pathlib import Path

import dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from utils import format_docs
from utils import split_sections, load_documents, flatten_sections, convert_chunks_to_documents

dotenv.load_dotenv()

vector_db_dir = '../data_chroma_multi'
collection_name = 'test_db'

if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=OpenAIEmbeddings(),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, root_title=title)
        _chunks = flatten_sections(sections)
        chunk_docs = convert_chunks_to_documents(_chunks)
        chunks.extend(chunk_docs)
    print(f"Split the documents into {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=vector_db_dir,
        collection_name=collection_name)

retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})

# query = "奥运火炬传递到达过珠峰吗?"
# query = "奥运会是什么时候开始停止支持4:3全屏转播的?"
# query = "中国在越野滑雪项目中的表现怎么样?"
query = "2024巴黎奥运会有棒球么?"
retrieved_docs = retriever.invoke(query)

# print(f"Retrieved {len(retrieved_docs)} documents")
# for doc in retrieved_docs:
#     print(f"Retrieved doc, {repr(doc.page_content)}")
#     # print(f"Retrieved doc meta, {doc.metadata}")
# exit(0)

llm = ChatOpenAI(model="gpt-4o-2024-08-06")
prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableParallel(
    contexts=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

result = rag_chain.invoke(query)
print(result)
print(result['answer'])

exit(0)
