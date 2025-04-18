from datetime import datetime


def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate a search query that is useful to retrieve related information
 from a collection of Wikipedia pages for answering the user query.

<USER_QUERY>
{user_query}
</USER_QUERY>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example user_query: 中国在奥运会上有哪些重要历史时刻?
Example output:
{{
    "query": "中国 奥运会 第一次",
    "rationale": "搜索中国在奥运会历史上的第一次重要时刻，例如首次参加、首次获奖等。"
}}
</EXAMPLE>

Provide your response in JSON format:"""

summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user query from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                              
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user query, skip it.                                                            
4. Ensure all additions are relevant to the user's query.                                                         
5. Verify that your final output differs from the input summary.
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary to ask the user's query: {user_query}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for search from a collection of Wikipedia pages. 

</REQUIREMENTS>

<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}

If you don't find any knowledge gaps, just say "No knowledge gaps found." in the knowledge_gap and return an empty string in the follow_up_query.

MUST RETURN the ``knowledge_gap`` and ``follow_up_query`` in Chinese!!!
</Task>

Provide your analysis in JSON format:"""
