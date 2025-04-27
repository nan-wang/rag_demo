from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# FakeEmbedding用于测试，随机生成向量
# pip install -U numpy
vector_store = InMemoryVectorStore(embedding=FakeEmbeddings(size=64))

document_1 = Document(
    page_content="今天天气很好",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="今天我去了公园",
    metadata={"source": "news"},
)

documents = [document_1, document_2]

vector_store.add_documents(documents=documents)

query = "今天天气怎么样?"
results = vector_store.similarity_search(query)
print(results)