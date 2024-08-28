import dotenv

dotenv.load_dotenv()

data = {
    'contexts': [[
        'article_title: 2020年夏季奥林匹克运动会\nsection_title: 转播商_Google涂鸦\ncontent: 2021年7月22日至8月8日，Google以奥运赛事为主题，在其首页的涂鸦上推出新的互动体育角色扮演游戏《冠军岛运动会》，以纪念日本文化、宣传竞技体育运动，以及向16位元日本经典电子游戏致敬。',
        'article_title: 2020年夏季奥林匹克运动会\nsection_title: 外部连结\ncontent: 官方网站 （日语、英语、法语、中文、韩语、西班牙语和印地语）\n东京2020年残奥运会的新浪微博（简体中文）\nTokyo 2020的X（前Twitter）帐户（日语）\nTokyo 2020的Facebook专页（英文）\nTokyo 2020的Instagram帐户（英文）\nTOKYO 2020的TikTok账户（日语）\nYouTube上的Tokyo 2020频道（日语）',
        'article_title: 2020年夏季奥林匹克运动会\nsection_title: 赛程_转播商\ncontent: 在日本，本届奥运会仍由NHK与民营广播业者组成的日本广播联合体转播。在台湾，本届奥运会由公视、东森电视以及爱尔达电视共同转播。在美国，本届奥运会将由NBC环球下属的有线电视频道NBC和串流媒体Peacock共同转播，其中NBC的转播权是从2014年冬季奥林匹克运动会开始生效的43.8亿美元协议的一部分，而Peacock的转播权则为2020年由NBC调整原有协议后随Peacock开始运营而生效。英国由BBC转播。中国大陆将由中国中央电视台转播实时赛况，中央人民广播电台中国之声频率也会对一些焦点赛事进行音频转播。香港播映权由特区政府购入，并交予本地电视台播放。',
        'article_title: 2020年夏季奥林匹克运动会\nsection_title: summary\ncontent: 2020年夏季奥林匹克运动会（日语：2020年夏季オリンピック／2020ねんかきオリンピック Nisen Nijū-nen Kaki Orinpikku ?；英语：2020 Summer Olympics），一般称为2020年东京奥运会、东京2020（Tokyo 2020，日语：东京2020／とうきょうニーゼロニーゼロ tōkyō nīzero-nīzero），又被称作复兴五轮（复兴五轮／ふっこうごりん Fukkō Gorin），于2021年7月23日至8月8日在日本东京都举行的第32届夏季奥林匹克运动会，为期17天。',
        'article_title: 2020年夏季奥林匹克运动会\nsection_title: 体育项目图标_海报\ncontent: 2020年1月7日，19名艺术家创作的20款艺术海报公布，其中12款为奥运会主题，9款为残奥会主题。'
    ],],
    'question': ['2020年夏季奥林匹克运动会的官方社交媒体在哪些平台上可以找到？',],
    'answer': ['2020年夏季奥林匹克运动会的官方社交媒体可以在以下平台上找到：新浪微博、X（前Twitter）、Facebook、Instagram、TikTok和YouTube。',],
    'ground_truth': ['2020年夏季奥林匹克运动会的官方社交媒体可以在新浪微博、Twitter（现称为X）、Facebook、Instagram、TikTok和YouTube上找到。',],
}

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_precision,
)
answer_relevancy.llm = None
context_recall.llm = None
context_precision.llm = None
answer_relevancy.embeddings = None
context_recall.embeddings = None
context_precision.embeddings = None

from datasets import Dataset

dataset = Dataset.from_dict(data)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-2024-08-06")
result = evaluate(
    dataset = dataset,
    metrics=[
        context_precision,
        context_recall,
        answer_relevancy,
    ],
    llm=llm,
    embeddings=OpenAIEmbeddings(),
)

df = result.to_pandas()

df.to_json("results.json", orient="records", indent=4, force_ascii=False)
