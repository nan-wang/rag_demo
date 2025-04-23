from pathlib import Path

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils import load_documents, get_chunks, format_docs

dotenv.load_dotenv()

vector_db_dir = '../data_chroma'
collection_name = 'test_db'

docs = load_documents("../data/*.txt")
print(f"Loaded {len(docs)} documents")

chunks = get_chunks(docs)
print(f"Split the documents into {len(chunks)} chunks")

if Path(vector_db_dir).exists():
    vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False, collection_name=collection_name)
    print(f"Loaded {vectorstore._chroma_collection.count()} documents")
else:
    # walk through the text files under "data" directory
    docs = load_documents("data/*.txt")
    print(f"Loaded {len(docs)} documents")

    chunks = get_chunks(docs)
    print(f"Split the documents into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=vector_db_dir,
        collection_name=collection_name)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

from langchain.prompts import ChatPromptTemplate

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-2024-08-06")

generate_queries = (
        prompt_rag_fusion
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads


def rrf(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        loads(doc) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


retrieval_chain_rag_fusion = generate_queries | retriever.map() | rrf

# result = generate_queries.invoke("奥运会的奖牌有什么环保设计?")

template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
        {"context": retrieval_chain_rag_fusion | format_docs, "question": RunnablePassthrough()}
        | RunnableParallel(
    contexts=RunnablePick("context"),
    question=RunnablePick("question"),
    answer=prompt | llm | StrOutputParser())
)

query = "奥运会的奖牌有什么环保设计?"
# result = retrieval_chain_rag_fusion.invoke(query)
result = rag_chain.invoke(query)
# for doc in result:
#     # print(f"{repr(r)}")
#     print(f"{repr(doc.page_content)}")
#     print(doc.metadata)
print(result)

{
    'contexts': 'article_title: 2008年夏季奥林匹克运动会\nsection_title: 筹备工作_奖牌\ncontent: 本届奥运会的奖牌于2007年3月27日公布，奖牌柔合了中西文化的特点：它的背面镶嵌了中国传统双龙蒲纹璜图案的玉壁，被称为「金镶玉」（金牌是白玉、银牌是青白玉、铜牌为青玉，昆仑玉质地），背面中央在金或银或铜质地的金属（采用智利金属）上镌刻著北京奥运会的会徽。这是夏季奥运会奖牌首次使用金属以外的物料制作。而正面为国际奥委会规定的希腊胜利女神和希腊帕那辛纳克体育场（Panathinaiko Stadium，首届现代奥运会举行场地）。由于过往奥运会奖牌在设计上没有太大的突破，因此此届奥运奖牌为奥运历史上带来重要的象征意义。奖牌设计师为中央美术学院设计学院、建筑学院党总支书记王沂蓬，奖牌最终在2008年3、4月份制作完成。\n北京奥委称，奖牌形象诠释了的表达了中华民族「以玉比德」的价值观。奖牌直径70毫米，厚6毫米。同时，奖牌的丝带由机织组成，朱地云纹。奖牌包装盒为中国传统工艺制作的木制漆盒，四方造型，天地盖四边略呈弧形，喻天地四方、六合美满之意。\n\narticle_title: 2012年夏季奥林匹克运动会\nsection_title: 吉祥物的故事_奖牌设计\ncontent: 2012年伦敦奥运奖牌由大卫·沃特金斯（David Watkins）所设计，于2011年10月终于亮相，于2011年11月开始制造，奖牌用美国犹他州跟蒙古沙漠的金银铜矿产打造，直径8.5公分，厚0.7公分，重量375公克到400公克之间，无论尺寸还是重量都是历届奥运之最，不过金牌其实只含有百分之1.2的黄金，本届奥运会将颁发302面金牌，再加上银牌跟铜牌，4700面奖牌总重量将近八公吨，整块金牌的价值约为650美元，银牌含有93%的银及7%的铜价值335美元，而铜牌价值不到5美元。\n巴西柔道选手费利佩·北代在洗澡的时候把奥运会铜牌弄坏了，事后，北代解释道：“我当时很害怕把它弄湿，所以把它咬在嘴里，但是突然一滑它就掉到地上了。”而在这块奖牌摔地之后，奖牌和带子之间的连接处摔坏了。之后巴西代表团团长伯纳德-拉加曼已经向国际奥委会提出了申请，希望能更换一块奖牌。\n\narticle_title: 2024年夏季奥林匹克运动会\nsection_title: 会徽_奖牌\ncontent: 2024年巴黎奥运奖牌由Chaumet进行设计、巴黎造币厂进行铸造。正面采用宝石镶嵌工艺嵌入一块过去翻修埃菲尔铁塔时保存的碎零件铁片。奖牌将负盛名的法国文化遗产与奥运融合，象征世界各国运动员享受荣耀，得以带回一部分的巴黎。铁片周围如光芒放射状的纹理，呼应巴黎「光之城」(La Ville-Lumière) 的美名。\n奖牌铁片借由处理恢复成埃菲尔铁塔原始颜色，切割成六边形后镶嵌在奖牌上。除了会徽铁片，整块奖牌的金银铜圆盘材质均采用回收金属制成；不仅资源再利用，也呼应环保理念。奖牌的背面为国际奥委会所规定的希腊胜利女神、帕那辛纳克体育场及雅典卫城图案，与过往几届夏季奥运不同的是，该届奥运奖牌的胜利女神面右侧多出了埃菲尔铁塔的图案。\n在奥运会奖牌未正式确定前，候选设计方案之一是由法国设计师菲利普·斯塔克设计，采用四合一的可拆分式设计，仅首枚奖牌连接绶带，其余三份奖牌可供运动员赠送他人留念。\n\narticle_title: 2010年冬季奥林匹克运动会\nsection_title: 大会标志_奖牌\ncontent: 本届冬奥会的奖牌是由当地的原居民亨特和设计师阿尔贝尔花18个月完成。这届赛事的奖牌最大的特色是呈波浪形设计，象征著主办国的波浪、山脉和雪，是在历届夏季奥运会或冬季奥运会中的首次。同时，本届冬奥会奖牌设计的另一特色是每面奖牌上的花纹图案亦不一样，也突破了历届奥运会的限制。另外，本届赛事的奖牌是众多届奥运会中最重的一面，约重500至576克。\n\narticle_title: 2018年冬季奥林匹克运动会\nsection_title: 赛程表_奖牌\ncontent: 奖牌设计体现韩国传统和文化，重量从金牌的586克到铜牌的493克不等。正面是奥林匹克五环和取材于树干的对角线，象征着运动员的决心。反面是大项、小项和平昌2018标志。共有259套奖牌被铸造。整套奖牌是韩国设计师李苏佑的作品，他将韩国字母和韩国文化的基础谚文融汇到他们的设计中，通过一系列的辅音象征着全世界运动员的付出，他们聚在一起，一齐参赛。缎带利用韩国传统织物制造，浅蓝绿色间浅红色相间，也采用了谚文字母的绣花。\n\narticle_title: 2020年夏季奥林匹克运动会\nsection_title: 口号_奖牌\ncontent: 奖牌由设计师川西俊一操刀，由公众捐赠的废旧电器提炼金属制成；挂带采用了棋盘图风格，配色沿用日本和服的重叠颜色（重ね の 色目）；外盒采用青梻制成，配色是奥运五色；盒子的盖子打开时如同奥运五环，由磁铁与外盒相连。奖牌正面印有希腊胜利女神尼刻、帕那辛纳克体育场和奥运五环。\n\n随著时间的推移，奥运会的结果和形象特别受到禁药的负面影响。在2016年俄罗斯兴奋剂丑闻之后，大量重新检查样品，导致了大量的取消资格事件发生，特别是在田径运动和举重方面。2022年3月时，一共有30面奖牌因为禁药问题遭到收回。\n\narticle_title: 2022年冬季奥林匹克运动会\nsection_title: 口号_奖牌\ncontent: 2022年冬奥的奖牌设计于2021年10月26日在北京2022年冬奥会倒计时100天主题活动仪式上发布。本届冬奥会奖牌由圆环加圆心构成牌体，形象来源于中国古代同心圆玉璧，共设五环，承袭了2008年夏季奥林匹克运动会的奖牌设计。奖牌正面中心刻有奥林匹克标志，周围刻有该届冬奥英文全称“XXIV Olympic Winter Games Beijing 2022”字样。打凹处理的圆环取意传统弦纹玉璧，装饰纹样来自中国传统纹样。奖牌背面中心刻有该届冬奥会会徽，周围刻有该届冬奥中文正式名称“北京2022年第24届冬季奥林匹克运动会”字样。圆环上的24个点及运动弧线取意古代天文图，象征着浩瀚星空、人与自然的和谐，及如群星璀璨的参赛运动员。奖牌背面最外环镌刻获奖运动员对应的比赛小项名称。',
    'question': '奥运会的奖牌有什么环保设计?',
    'answer': '奥运会的奖牌中，2024年巴黎奥运会的奖牌采用了环保设计，整块奖牌的金银铜圆盘材质均使用回收金属制成，这不仅是资源再利用，也呼应环保理念。此外，2020年东京奥运会的奖牌也是由公众捐赠的废旧电器中提炼金属制成的。'}
