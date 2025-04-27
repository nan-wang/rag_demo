from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models import ParrotFakeChatModel
from time import time


rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.25,  # 每4秒发送一个请求
    check_every_n_seconds=10,  # 每个0.1秒检查一次是否可以继续发送新的请求
    max_bucket_size=2,  # 等待队列最多有2个请求
)


# 调用硅基流动提供的DeepSeek-R1模型
llm = ParrotFakeChatModel(rate_limiter=rate_limiter)

for _ in range(5):
    tic = time()
    llm.invoke("你好")
    toc = time()
    print(f"距离上一次请求: {toc - tic:.3f}秒")
# 输出：
# 0s: 可发送请求数=0,
# 4s: 等待队列[1]
# 8s: 等待队列[1, 1]
# 10s: 检查等待队列，更新可发送请求数=2，可发送请求数>0, 发送请求，距离上一次请求: 10.031秒, 可发送请求数=1，等待队列[1]
# 10s: 可发送请求数>0, 发送请求，距离上一次请求: 0.002秒, 可发送请求数=0，等待队列[]
# 12s: 等待队列[1]
# 16s: 等待队列[1, 1]
# 20s: 等待队列[1, 1]达到上限，不再新增
# 20s: 检查等待队列，更新可发送请求数=2，距离上一次请求: 10.004秒, 可发送请求数=1，等待队列[1]
# 20s: 可发送请求数>0, 发送请求，距离上一次请求: 0.002秒, 可发送请求数=0，等待队列[]
# 24s: 等待队列[1]
# 28s: 等待队列[1, 1]
# 30s: 检查等待队列，更新可发送请求数=2，距离上一次请求: 10.004秒, 可发送请求数=1，等待队列[1]