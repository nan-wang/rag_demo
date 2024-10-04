import sys
import json

import dotenv
from pathlib import Path
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_documents, get_chunks, format_docs, split_contexts

dotenv.load_dotenv()

vector_db_dir = '../data_chroma_with_metadata'
collection_name = 'test_db'
data_path = "../data/*.txt"


if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(), create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents(data_path, with_metadata=True)
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir, collection_name=collection_name)


from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

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

llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    search_kwargs={"k": 5},
)

query = "2020奥运会有哪些兴奋剂相关新闻?"

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

"""
{
    'contexts': '俄罗斯因体育禁药事件遭禁赛四年，但俄罗斯已就本次决定提起上诉，俄罗斯已于2019年12月31日对此决定向国际体育仲裁院提出上诉。2020年12月17日，国际体育仲裁法庭裁定俄罗斯反兴奋剂机构违规，决定对俄罗斯反兴奋剂中心处以为期两年的处罚。期间俄罗斯无法参加包括2020年东京奥运会等大型国际体育赛事。运动员可以以个人名义参赛，本次奥运俄罗斯选手受禁令不能以国家代表队的名义参赛，并不得使用国号、国旗、国歌及国家编码「RUS」，只能以俄罗斯奥林匹克委员会名义参与，代表队编码为「ROC」（俄国奥会的英语缩写），旗帜为俄国奥会会旗。\n澳洲、加拿大和英国在奥运宣布延期前一度宣布因应2019冠状病毒病疫情，不会参加未延期的东京奥运。\n\n2021年7月16日，东京奥组委公布了防疫准则，要求各国运动员禁止握手或拥抱，以避免不必要的肢体接触，甚至严禁选手们性交。但这项政策却引发部分选手不满。年届52岁的德国前跳远名将苏森·蒂德克即表示，「选手们以『这种方式』发泄精力是很正常的事情」。疫情下的东京奥运估计将有11000位运动员参加，各国选手齐聚于奥运盛会除了在赛场上拼成绩、为国家争光外，在选手村里彼此交流也向来相当热络。据东京奥组委表示，由于全球疫情尚未消退，本届东京奥运为了避免爆发群聚感染，严禁运动员在奥运期间发生性行为，但仍依照过往惯例，提供选手村15万个安全套，并告知选手这些安全套可作为纪念品，相比上届的里约奥运，本届奥运发放的安全套数量足足少了29万个。此外，本届东奥讲求环保，选手村内的床架首创以厚纸板制成。\n\narticle_title: 2020年夏季奥林匹克运动会\nsection_title: 转播商_Google涂鸦\ncontent: 2021年7月22日至8月8日，Google以奥运赛事为主题，在其首页的涂鸦上推出新的互动体育角色扮演游戏《冠军岛运动会》，以纪念日本文化、宣传竞技体育运动，以及向16位元日本经典电子游戏致敬。\n\narticle_title: 2020年夏季奥林匹克运动会\nsection_title: 志愿者_安保工作\ncontent: 2018年12月，东京都政府宣布禁止无人机在奥运会场馆上空飞行。\n\narticle_title: 2020年夏季奥林匹克运动会\nsection_title: 圣火传递_比赛项目\ncontent: 包含28种原有项目，加上5种新项目，分别是空手道、滑板、运动攀登、冲浪以及棒垒球。空手道和棒垒球只会在本届出现，下届将会取消，其余三项则保留，被取消的项目将由霹雳舞取代以吸引年轻观众。',
    'question': '2020奥运会有哪些兴奋剂相关新闻?',
    'answer': '在2020年东京奥运会上，与兴奋剂相关的新闻包括俄罗斯因禁药事件被禁赛四年。裁决允许俄罗斯运动员以个人名义参赛，但不能使用国家标志，只能以俄罗斯奥林匹克委员会名义参赛。国际体育仲裁法庭对俄罗斯反兴奋剂机构进行了两年的处罚。'}
"""
