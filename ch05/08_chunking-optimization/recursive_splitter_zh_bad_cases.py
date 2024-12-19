from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = """
第三十届夏季奥林匹克运动会（英语：the Games of the XXX Olympiad；法语：les Jeux de la XXXe Olympiade），一般称为2012伦敦奥运会，是于2012年7月27日至8月12日在英国伦敦举行的一届综合性运动会。第一项赛事，女足预赛在开幕式前两天7月25日已经开始。205个国家和地区奥委会的1万多名运动员参加比赛。\n2005年7月6日，国际奥委会在新加坡举行的第117次国际奥委会会议上宣布，由伦敦主办此届奥运会，亦是继1908年和1948年后，伦敦第三次取得夏季奥运举办权，也使伦敦成为至今举办奥运会次数最多的城市。本次奥运也是英国女王伊莉莎白二世继1976年夏季奥林匹克运动会后第二次宣布奥运开幕，也使其成为史上唯一两次宣布夏季奥运开幕的国家元首。\n随著时间的推移，奥运会的结果和形象特别受到禁药的负面影响。在2016年俄罗斯兴奋剂丑闻之后，大量重新检查样品，导致了大量的取消资格事件发生，特别是在田径运动和举重方面。2022年3月时，一共有30面奖牌因为禁药问题遭到收回。
"""

doc_1 = Document(page_content=text, metadata={"doc_id": 1})


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    add_start_index=True,
)


all_splits = text_splitter.split_documents([doc_1, ])

print(f"Split the documents into {len(all_splits)} chunks")

for i, split in enumerate(all_splits):
    print(f"Split {i} {repr(split.page_content)}")