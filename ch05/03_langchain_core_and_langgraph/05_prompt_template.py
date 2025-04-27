from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("{location}的天气怎么样？")

prompt = prompt_template.invoke({"location": "北京"})

print(prompt)
print(type(prompt))