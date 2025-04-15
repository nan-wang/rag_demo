from typing import Any, Dict, List, Union
from tavily import TavilyClient
import re
import json


def get_config_value(value: Any) -> str:
    return value if isinstance(value, str) else value.value


def strip_thinking_tokens(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text


def tavily_search(query: str, fetch_full_page: bool = True, max_results: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    client = TavilyClient()
    return client.search(
        query,
        max_results=max_results,
        include_full_page=fetch_full_page)


def deduplicate_and_format_sources(
        search_response: Union[Dict[str, Any], List[Dict[str, Any]]],
        max_tokens_per_source: int,
        fetch_full_page: bool = True) -> str:
    if isinstance(search_response, dict):
        sources_list = search_response["results"]
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and "results" in response:
                sources_list.extend(response["results"])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Invalid search response format")

    unique_sources = {}
    for source in sources_list:
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
    return formatted_text.strip()


def format_sources(search_results: Dict[str, Any]) -> str:
    if not search_results:
        return ""
    return "\n".join(
        f"* {source['title']}: {source['url']}" for source in search_results["results"]
    )


def extract_json_from_markdown(input_str):
    match = re.search(r'```json\s*(\{.*?\})\s*```', input_str, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    return input_str