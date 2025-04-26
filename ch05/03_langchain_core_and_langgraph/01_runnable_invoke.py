from langchain_core.runnables import RunnableGenerator


def gen_regards(input):
    for token in ["Have", " a", " nice", " day"]:
        yield token


runnable = RunnableGenerator(gen_regards)
# invoke方法
response = runnable.invoke(None)
print(response)
# 输出
# Have a nice day
