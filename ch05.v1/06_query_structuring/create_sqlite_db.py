# https://python.langchain.com/docs/how_to/sql_csv/#sql
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path

db_name = "olympic_games"
engine = create_engine(f"sqlite:///{db_name}.db")

if not Path(f"{db_name}.db").exists():
    # create db
    data_sources = (
        "../data_sql/olympic_hosts.csv",
        "../data_sql/olympic_medals.csv"
    )
    for data_fn in data_sources:
        table_name = Path(data_fn).stem.split("_")[-1]
        df = pd.read_csv(f"{data_fn}")
        print(table_name)
        print(df.shape)
        print(df.columns.tolist())
        df.to_sql(f"{table_name}", engine, index=False)

db = SQLDatabase(engine=engine)
tables = db.get_usable_table_names()
print(tables)


