from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt_template = ChatPromptTemplate(
    [("system", "你是一位乐于帮助的助理"), MessagesPlaceholder("msgs")]
)

result = prompt_template.invoke(
    {
        "msgs": [
            HumanMessage(content="我是来自上海的开发者"),
            AIMessage(content="你好，今天有什么可以帮助你的?"),
        ]
    }
)

print(type(result))
print(result)