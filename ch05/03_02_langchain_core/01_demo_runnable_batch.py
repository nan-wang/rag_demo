from langchain_core.runnables import RunnableGenerator


def gen_regards(input):
    for token in ["Have", " a", " nice", " day"]:
        yield token


runnable = RunnableGenerator(gen_regards)
# batch方法
response = runnable.batch([None, None])
print(response)
# 输出: ['Have a nice day', 'Have a nice day']
