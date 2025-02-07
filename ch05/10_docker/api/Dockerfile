# chatbot_api/Dockerfile

# FROM python:3.11-slim
FROM alibaba-cloud-linux-3-registry.cn-hangzhou.cr.aliyuncs.com/alinux3/python:3.11.1

WORKDIR /app
COPY ./src /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install /code/.

EXPOSE 8000
CMD ["sh", "entrypoint.sh"]