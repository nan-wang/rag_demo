import os
from typing import Any, Optional, Literal

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    max_web_search_results: int = Field(
        default=1,
        title="Max Web Search Results",
        description="Maximum number of web search results to return"
    )
    max_web_search_loops: int = Field(
        default=3,
        title="Search Depth",
        description="Number of search iterations to perform"
    )
    reasoning_llm: str = Field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        title="LLM Model Name",
        description="Name of the LLM model to use"
    )
    search_api: Literal["tavily", "duckduckgo"] = Field(
        default="tavily",
        title="Search API",
        description="Web search API to use"
    )
    fetch_full_page: bool = Field(
        default=True,
        title="Fetch Full Page",
        description="Include the full page content in the search results"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.configurable if hasattr(config, 'configurable') and config else {}

        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
