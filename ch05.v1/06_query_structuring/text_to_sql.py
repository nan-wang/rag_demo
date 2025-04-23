import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePick
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

db_name = "olympic_games"
db = SQLDatabase.from_uri(f"sqlite:///{db_name}.db")

# https://python.langchain.com/docs/tutorials/sql_qa/#chains

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

prompt_str = """You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Question: Question here
SQLQuery: SQL Query to run. Don't use the sql markdown grammar.

Only use the following tables:
{table_info}

Don't start the reply with `SQLQuery: `!!!

Question: {input}
"""
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(prompt_str)
execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db, prompt=prompt)

answer_prompt_str = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: 
"""

answer_prompt = PromptTemplate.from_template(answer_prompt_str)

chain = (
        RunnableParallel(
            question=RunnablePick("question"),
            query=write_query)
        | RunnableParallel(
            question=RunnablePick("question"),
            query=RunnablePick("query"),
            result=RunnablePick("query") | execute_query)
        | answer_prompt
        | llm
        | StrOutputParser()
)
# chain = (
#     RunnablePassthrough.assign(query=write_query).assign(
#         result=itemgetter("query") | execute_query
#     )
#     | answer_prompt
#     | llm
#     | StrOutputParser()
# )
result = chain.invoke({"question": "中国队在巴黎奥运会上有多少运动员获得金牌?"})
