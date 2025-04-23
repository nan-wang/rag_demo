import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/ask")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        基于LangChain的奥运会问答机器人，可以解答关于1984年到2024年期间历届奥运会的问题。 
        """
    )

    st.header("你可以这样问")
    st.markdown("- 巴黎奥运会的吉祥物是什么?")
    st.markdown("- 奥运会上有哪些环保措施?")
    st.markdown("- 北京是在哪一年成功申办冬季奥运会的？")
    st.markdown("- 在2022年北京冬奥会中，通过哪些平台观看的人数创下了多少纪录？")
    st.markdown("- Google推出的与2020年奥运会相关的游戏名称是什么？")
    st.markdown("- 在2010年冬季奥运会上，哪个国家获得了最多金牌？")
    st.markdown("- 2010年冬季奥林匹克运动会的奖牌有什么特别的设计特点？")
    st.markdown("- 2018年冬季奥林匹克运动会中，俄罗斯奥委会因何原因被禁止参加比赛？")
    st.markdown("- 2024年夏季奥林匹克运动会中，马拉松项目是否会开放给公众参与？")
    st.markdown("- 1984年夏季奥运会上，卡尔·刘易斯获得了多少枚金牌？")
    st.markdown("- 2022年冬奥会计划招募多少名赛会志愿者？")
    st.markdown("- 2010年冬季奥林匹克运动会在哪些地方举行？")
    st.markdown("- 在2020年夏季奥林匹克运动会期间，国立竞技场被称为什么名称？")
    st.markdown("- 2014年冬季奥林匹克运动会的圣火采集仪式是在何时何地举行的？")
    st.markdown("- 2022年冬奥会新增了哪些比赛项目？")
    st.markdown("- 2014年冬季奥运会的火炬是由谁点燃的？")
    st.markdown("- 2016年夏季奥运会的国内转播权是卖给了哪个集团？")

st.title("奥运会问答机器人")
st.info("可以问我关于1984年到2024年期间历届奥运会的问题")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "output": "你想了解关于奥运会的什么知识？我知道关于1984年到2024年期间历届奥运会的很多信息",
            "explanation": "N/A"
        },
    ]

import re

def get_explanation(text):
  """
  Extracts article title, section title, and content from a text using regex.

  Args:
    text: The input text in the format "[doc_X]article_title: ... section_title: ... content: ...".

  Returns:
    A dictionary containing the extracted article_title, section_title, and content.
    Returns None if the input format is invalid.
  """
  pattern = r"\[doc_\d+\]article_title:\s*(.*?)\s*section_title:\s*(.*?)\s*content:\s*(.*)"
  match = re.search(pattern, text)

  if match:
    article_title = match.group(1).strip()
    section_title = match.group(2).strip()
    content = match.group(3).strip()
    return article_title, section_title, content
  else:
    return None


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("参考信息", state="complete"):
                article_title = ""
                section_title = ""
                context = ""
                result = get_explanation(message["explanation"])
                if result is not None:
                    article_title, section_title, context = result
                st.info(f"文章来源：{article_title}")
                st.info(f"文章章节：{section_title}")
                st.info(f"文章内容：{context}")

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["answer"]
            explanation = response.json()["selected_content"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("参考信息", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )