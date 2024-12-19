multi_context_prompts = """
The task is to write a  question in a way that answering it requires information derived from both context1 and context2. 
    Follow the rules given below while writing the question.
        1. The question should not be very long. Use abbreviation wherever possible.
        2. The question must be reasonable and must be understood and responded by humans.
        3. The question must be fully answerable from information present in context1 and context2. 
        4. Read and understand both contexts and write the question so that answering requires insight from both context1 and context2.
        5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question. 
        6. The question should be a single question rather than multiple questions to answer.

Return in Chinese!!!

here is two examples,

context1: "article_title: 1996年夏季奥利匹克奥运\nsection_title: 比赛项目\ncontent: 本届奥运会共26个大项目、271个小项目。",
context2: "article_title: 2000年夏季奥利匹克奥运\nsection_title: 比赛项目\ncontent: 本届奥运会共28个大项300个小项。",
question: "1996年夏季奥运和2000年夏季奥运的比赛项目总数分别是多少？",
```

Your actual task:

context1: "article_title: 2024年夏季奥利匹克奥运\nsection_title: 吉祥物\ncontent: 巴黎奥运会的吉祥物为奥林匹克弗里热（Phryges），灵感源自于弗里吉亚无边便帽，为法国大革命的象征女神玛丽安娜所戴的帽子，代表法兰西共和国的其中一个象征 - 自由。"。
context1: "article_title: 2016年夏季奥利匹克奥运\nsection_title: 吉祥物\ncontent: 奥运会吉祥物「费尼希斯」的名称来自巴西音乐家费尼希斯·迪·摩赖斯，它的设计中带有哺乳动物的特点，象征巴西的野生动物。吉祥物的虚构背景故事表明它们是由巴西人在里约被宣布为东道主后的喜悦之情孕育的。品牌总监贝丝·卢拉（Beth Lula）表示吉祥物旨在反应巴西文化和人民的多样性。吉祥物名字凭借占全票数44%的323,327票战胜另外两组名字，其结果于2014年12月14日公布。"。
"""