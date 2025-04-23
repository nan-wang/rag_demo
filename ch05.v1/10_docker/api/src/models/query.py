from pydantic import BaseModel


class QueryInput(BaseModel):
    text: str


class QueryOutput(BaseModel):
    selected_content: str
    answer: str
