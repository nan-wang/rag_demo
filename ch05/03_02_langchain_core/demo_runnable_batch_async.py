import asyncio

from langchain_core.runnables import RunnableGenerator


async def agen_regards(input):
    for token in ["Have", " a", " nice", " day"]:
        yield token


async def main():
    runnable = RunnableGenerator(agen_regards)
    response = await runnable.abatch([None, None])
    print(response)
    # 输出：
    # ['Have a nice day', 'Have a nice day']


if __name__ == "__main__":
    asyncio.run(main())
