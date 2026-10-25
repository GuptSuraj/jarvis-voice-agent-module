import logging
from livekit.agents import function_tool, RunContext
from langchain_community.tools import DuckDuckGoSearchRun

@function_tool
async def search_web(context: RunContext, query:str):
    """
    Use this tool to search the web for information related to the given query.
    """

    try:
        result = DuckDuckGoSearchRun().run(tool_input=query)
        logging.info(f"Web search result for '{query}': {result}")
        return result
    except Exception as e:
        logging.error(f"Error occurred while searching on web for '{query}': {e}")
        raise

