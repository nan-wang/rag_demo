from langchain_openai.chat_models import ChatOpenAI

# 调用OpenAI提供的gpt-4o-mini模型
llm = ChatOpenAI(model="gpt-4o-mini", api_key="sk-<YOUR_OPENAI_API_KEY>")
response = llm.invoke("你好！")
print(response.content)
# 输出：'你好！今天过得怎么样？有什么我可以帮助你的吗？'

# 调用硅基流动提供的DeepSeek-R1模型
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    api_key="sk-<YOUR_SILICONFLOW_API_KEY>",
    base_url="https://api.siliconflow.cn/v1",
)
response = llm.invoke("你好！")
print(response.content)
# 输出：'你好！很高兴见到你，有什么我可以帮忙的吗？'