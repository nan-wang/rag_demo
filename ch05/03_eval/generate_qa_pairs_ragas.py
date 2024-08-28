import dotenv

from utils import load_documents, get_chunks
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()

docs = load_documents("data/2024年夏季奥林匹克运动会.txt")
documents = get_chunks(docs)
for d in documents:
    d.metadata['filename'] = d.metadata['source']


# generator with openai models
generator_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
critic_llm = ChatOpenAI(model="gpt-4o-2024-08-06")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={multi_context: 1.}, with_debugging_logs=True)

testset.to_pandas().to_json("ragas_testset.json", force_ascii=False, indent=4)