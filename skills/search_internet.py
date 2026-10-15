import os
from tavily import TavilyClient

api_key = os.environ.get("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required.")

tavily_client = TavilyClient(api_key=api_key)
response = tavily_client.search("Who is Leo Messi?")

print(response)