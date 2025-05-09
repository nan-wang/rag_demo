FROM python:3.11-slim

WORKDIR /app

COPY ./src/ /app

COPY ./pyproject.toml /code/pyproject.toml
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install /code/.

CMD ["sh", "entrypoint.sh"]