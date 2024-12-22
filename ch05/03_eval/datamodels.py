from langchain_core.pydantic_v1 import BaseModel, Field


class KeyPoint(BaseModel):
    question: str = Field(..., description="The question.")
    answer: str = Field(..., description="The answer.")
    keypoint: str = Field(..., description="The keypoint related to the question which should be covered by the answer")
    label: str = Field("Relevant", description="The label indicating whether the answer covers the keypoint.")