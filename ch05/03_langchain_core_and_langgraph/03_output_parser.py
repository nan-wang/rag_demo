import dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()

model = OpenAI(model="deepseek-ai/DeepSeek-V3")

# 定义期望的输出格式
class Joke(BaseModel):
    setup: str = Field(description="冷笑话的问题")
    punchline: str = Field(description="冷笑话的答案")


# 构造JsonOutputParser
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="回答用户的请求。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser
output = chain.invoke({"query": "给我讲个冷笑话"})
print(output)
# 输出： {'setup': '你知道为什么电脑很冷吗？', 'punchline': '因为它有很多风扇。'}