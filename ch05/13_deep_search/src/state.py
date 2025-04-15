import operator
from dataclasses import dataclass, field
from typing_extensions import Annotated


@dataclass(kw_only=True)
class SummaryState:
    search_topic: str = field(default=None)  # search topic
    search_query: str = field(default=None)  # search query
    web_search_results: Annotated[list, operator.add] = field(default_factory=list)  # web search results
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    search_loop_count: int = field(default=0)  # search loop count
    running_summary: str = field(default=None)  # final report


@dataclass(kw_only=True)
class SummaryStateInput:
    search_topic: str = field(default=None)


@dataclass(kw_only=True)
class SummaryStateOutput:
    running_summary: str = field(default=None)