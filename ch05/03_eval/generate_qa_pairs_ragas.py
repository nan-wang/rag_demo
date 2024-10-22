import dotenv
# pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.docstore import InMemoryDocumentStore
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, multi_context, reasoning
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

vector_db_dir = '../data_chroma_multi'
collection_name = 'test_db'
num_docs = 10

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
import random

documents = []
ids = vectorstore.get()['ids']
random.shuffle(ids)
for id in ids[:num_docs*2]:
    doc = vectorstore.get(id)
    if not doc["metadatas"][0]["is_leaf"]:
        continue
    documents.append(Document(page_content=doc["documents"][0], metadata=doc["metadatas"][0], id=doc["ids"][0]))

# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
critic_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator_llm_model = LangchainLLMWrapper(generator_llm)
critic_llm_model = LangchainLLMWrapper(critic_llm)
embeddings_model = LangchainEmbeddingsWrapper(embeddings)

keyphrase_extractor = KeyphraseExtractor(llm=generator_llm_model)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    add_start_index=True,
    separators=['。', '！', '？', '\?', '\n\n', '\n', '\n\n\n'],
    is_separator_regex=True,
    keep_separator="end"
)
keyphrase_extractor.adapt(language="chinese")
docstore = InMemoryDocumentStore(
    splitter=splitter,
    embeddings=embeddings_model,
    extractor=keyphrase_extractor,
    run_config=RunConfig(),
)

docstore.add_documents(documents)

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings,
    docstore=docstore
)

distributions = {
    simple: 1.0,
}

generator.adapt(language="chinese", evolutions=[simple, reasoning, multi_context])
# generator.save(evolutions=[simple, multi_context, reasoning])

# generate testset
testset = generator.generate(test_size=num_docs, distributions=distributions, with_debugging_logs=True)

testset.to_pandas().to_json("ragas_testset.1011.part3.json", force_ascii=False, indent=4)
