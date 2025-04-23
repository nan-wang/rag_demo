import json
from pathlib import Path

import dotenv
import pkuseg
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors.jina_rerank import JinaRerank
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from langchain_core.pydantic_v1 import BaseModel, Field
from utils import load_documents, split_sections, split_chunks, format_docs

dotenv.load_dotenv()


class Response(BaseModel):
    selected_content: str = Field(..., description="selected content from the context that is useful to answer the question.")
    answer: str = Field(..., description="the final answer to the question.")

def get_all_splits():
    docs = load_documents("../data/*.txt")
    print(f"Loaded {len(docs)} documents")
    chunks = []
    for doc in docs:
        text = doc.page_content
        article_title = Path(doc.metadata.get("source", "")).stem
        sections = split_sections(text, source=article_title)
        _chunks = split_chunks(sections)
        chunks.extend(_chunks)
    return chunks


def tokenize_doc(doc_str: str):
    result = []
    for l in doc_str.splitlines():
        ll = l.strip()
        if not ll:
            continue
        split_tokens = [t.strip() for t in seg.cut(ll) if t.strip() != '']
        result += split_tokens
    return result


vector_db_dir = '../data_chroma_jina_embeddings'
collection_name = 'olympic_games'
if Path(vector_db_dir).exists():
    vectorstore = Chroma(
        persist_directory=vector_db_dir,
        embedding_function=JinaEmbeddings(model_name="jina-embeddings-v3"),
        create_collection_if_not_exists=False,
        collection_name=collection_name)
    print(f"{vectorstore._chroma_collection.count()} documents loaded")
else:
    chunks = get_all_splits()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=JinaEmbeddings(model_name="jina-embeddings-v3"),
        persist_directory=vector_db_dir,
        collection_name=collection_name)

vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# set up the bm25 retriever
seg = pkuseg.pkuseg()

chunks = get_all_splits()
bm25_retriever = BM25Retriever.from_documents(
    chunks, preprocess_func=tokenize_doc)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

compressor = JinaRerank(model="jina-reranker-v2-base-multilingual", top_n=10)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Response)
prompt = ChatPromptTemplate.from_template(
    """You're a helpful AI assistant. Given a user question related to the Olympic Games and some Wikipedia article snippets, answer the user question and provide citations. If none of the articles answer the question, just say you don't know. 
Follow the steps,
Step 1: Read the ``Question``.
Step 2: Select the content useful to answer the ``Question`` from ``Context``.
Step 3: Use the selected content from Step 2 to generate an answer.
Use three sentences maximum and keep the answer concise.
------
举例:
Question: "中国在奥运会上有哪些重要历史时刻? "
Context: "[doc_1]article_title: 1996年夏季奥林匹克运动会\nsection_title: 焦点 香港为最后一次以「香港」和「Hong Kong」名义出席奥林匹克运动会，滑浪风帆选手李丽珊赢得香港历史性首面奥运金牌。\n\n[doc_1]article_title: 1984年夏季奥林匹克运动会 section_title: 焦点（社会主义国家里中国、罗马尼亚、南斯拉夫、索马里、贝宁、刚果和莫桑比克参加，这些国家与苏联关系较差） 中华人民共和国自1952年部份参加后，首次全程参与夏季奥运会，许海峰获得了中国也是本届奥运会的首枚金牌，实现了中国零的突破。\n\n[doc_2]article_title: 2002年冬季奥林匹克运动会 section_title: 焦点 本届奥运的开幕式比照1992年巴塞隆纳奥运，将开幕式从白天改至晚上举行。 中国在短道速滑女子500米决赛中，杨扬击败了保加利亚的艾芙金妮亚·拉达诺娃和队友王春露，夺得了冠军，为中国自1980年冬季奥林匹克运动会参赛以来首枚金牌。\n\n[doc_3]article_title: 1992年夏季奥林匹克运动会 section_title: 焦点 白俄罗斯的体操选手维塔里·谢尔博独自夺得6枚金牌，创下在单届奥运会中取得最多金牌的记录。 棒球首次成为正式奥运会项目，古巴夺得金牌，中国台湾夺得银牌。\n\n[doc_4]article_title: 2008年夏季奥林匹克运动会 section_title: summary\n主办国中华人民共和国以51面金牌成为金牌榜首名，是奥运历史上首个登上金牌榜首的亚洲国家，强化了中国作为体育强国的地位。美国以112面奖牌（36金，39银，37铜）为本届奥运会最多奖牌的国家。"
selected_content: "[doc_1]article_title: 1984年夏季奥林匹克运动会 section_title: 焦点 中华人民共和国自1952年部份参加后，首次全程参与夏季奥运会，许海峰获得了中国也是本届奥运会的首枚金牌，实现了中国零的突破。\n\n[doc_2]article_title: 2002年冬季奥林匹克运动会 section_title: 焦点 中国在短道速滑女子500米决赛中，杨扬击败了保加利亚的艾芙金妮亚·拉达诺娃和队友王春露，夺得了冠军，为中国自1980年冬季奥林匹克运动会参赛以来首枚金牌。\n\n[doc_4]article_title: 2008年夏季奥林匹克运动会 section_title: summary\n主办国中华人民共和国以51面金牌成为金牌榜首名，是奥运历史上首个登上金牌榜首的亚洲国家，强化了中国作为体育强国的地位。"
answer: "中国在奥运会上有几个重要的历史时刻。1984年，中华人民共和国首次全程参加夏季奥运会，并由许海峰赢得首枚金牌。2002年，中国选手杨扬在冬季奥运会上夺得首枚金牌。2008年，北京奥运会上，中国作为主办国首次登上金牌榜首，确立了其体育强国地位。"
------
以JSON格式返回结果。JSON对象必须包含以下键：
- 'selected_content'：selected content from the ``Context`` that is useful to answer the ``Question``
- 'answer': the final answer to the ``Question``.

下面是你的任务:

Question: {question} 
Context: {context}
"""
)

rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough())
        | RunnableParallel(
    context=RunnablePick("context"),
    question=RunnablePick("question"),
    response=prompt | llm)
)

results = []
with open("../03_eval/data_eval/v20241219/qa_pairs_rewrite.json", "r") as f:
    qa_pairs = json.load(f)
    for doc in tqdm(qa_pairs):
        query = doc["query"]
        try:
            result = rag_chain.invoke(query)
            doc["response"] = {
                "content": result["response"].answer,
                "contexts": [result["context"],],
                "selected_content": result["response"].selected_content,
            }
            results.append(doc)
        except Exception as e:
            print(f"Error: {e}")
            continue

output_path = "../03_eval/data_metrics/v20241219/ch0507_prompt/response.json"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
