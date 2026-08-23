from tavily import TavilyClient
# just use this API key because this API key do not important, you can use your own API key
tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("Who is Leo Messi?")

print(response)