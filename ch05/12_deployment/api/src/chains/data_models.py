from pydantic import BaseModel, Field


class Response(BaseModel):
    selected_content: str = Field(..., description="selected content from the context that is useful to answer the question.")
    answer: str = Field(..., description="the final answer to the question.")
