import dotenv
from langchain_openai.chat_models import ChatOpenAI
from pprint import pprint

dotenv.load_dotenv()

# 调用硅基流动提供的DeepSeek-R1模型
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    # api_key="sk-<YOUR_SILICONFLOW_API_KEY>",
    # base_url="https://api.siliconflow.cn/v1",
)
response = llm.invoke("你好！")
pprint(response.to_json())
# 输出：
# {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
#  'kwargs': {'additional_kwargs': {'refusal': None},
#             'content': '\n'
#                        '\n'
#                        '您好！很高兴为您提供帮助。您有什么问题或需要咨询的吗？无论是学习、生活还是其他方面的问题，我都在这里为您解答。请随时告诉我您的需求，我会尽力提供详细的回答和建议。',
#             'id': 'run-b7c4be28-0655-4af5-8416-fb07831299cf-0',
#             'invalid_tool_calls': [],
#             'response_metadata': {'finish_reason': None,
#                                   'logprobs': None,
#                                   'model_name': 'deepseek-ai/DeepSeek-R1',
#                                   'system_fingerprint': '',
#                                   'token_usage': {'completion_tokens': 88,
#                                                   'completion_tokens_details': {'accepted_prediction_tokens': None,
#                                                                                 'audio_tokens': None,
#                                                                                 'reasoning_tokens': 44,
#                                                                                 'rejected_prediction_tokens': None},
#                                                   'prompt_tokens': 7,
#                                                   'prompt_tokens_details': None,
#                                                   'total_tokens': 95}},
#             'tool_calls': [],
#             'type': 'ai',
#             'usage_metadata': {'input_token_details': {},
#                                'input_tokens': 7,
#                                'output_token_details': {'reasoning': 44},
#                                'output_tokens': 88,
#                                'total_tokens': 95}},
#  'lc': 1,
#  'type': 'constructor'}