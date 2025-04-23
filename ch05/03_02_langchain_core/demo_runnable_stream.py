from langchain_core.runnables import RunnableGenerator


# stream方法
def gen_regards(input):
    for token in ["Have", " a", " nice", " day"]:
        yield token


runnable = RunnableGenerator(gen_regards)
for c in runnable.stream(None):
    print(f"{c}|")
# 输出:
# Have|
# a|
# nice|
# day|
