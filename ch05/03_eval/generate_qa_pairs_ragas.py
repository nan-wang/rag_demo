import dotenv

from utils import load_documents, get_chunks
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

vector_db_dir = '../data_chroma'
collection_name = 'test_db'

from langchain_chroma import Chroma
from pathlib import Path
if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=OpenAIEmbeddings(),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
from langchain_core.documents import Document
documents = []
for id in vectorstore.get()["ids"]:
    doc = vectorstore.get(id)
    documents.append(Document(page_content=doc["documents"][0], metadata=doc["metadatas"][0], id=doc["ids"][0]))


# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
critic_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

distributions = {
    simple: 0.5,
    multi_context: 0.4,
    reasoning: 0.1
}

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=20, distributions=distributions)

testset.to_pandas().to_json("ragas_testset.1008.json", force_ascii=False, indent=4)