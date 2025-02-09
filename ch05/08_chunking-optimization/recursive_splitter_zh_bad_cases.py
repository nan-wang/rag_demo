from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

text = """
=== 赛事焦点 ===

京奥闭幕式后，不少传媒选出了本届奥运的赛事焦点，以下为路透社所选出的10大赛事焦点：

牙买加人博尔特于男子100米赛事以9秒69的成绩刷新世界纪录。
美国泳手菲尔普斯一圆八金梦，打破史毕兹在一届奥运中夺得7面金牌的最高纪录。
雅典奥运男子110米栏金牌得主刘翔因伤退出，无缘卫冕。
"""

doc_1 = Document(page_content=text, metadata={"doc_id": 1})


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,
    chunk_overlap=32,
    add_start_index=True,
)


all_splits = text_splitter.split_documents([doc_1, ])

print(f"Split the documents into {len(all_splits)} chunks")

for i, split in enumerate(all_splits):
    print(f"Split {i} {repr(split.page_content)}")