"""Shared NVIDIA Nemotron + Tavily client helpers used by the multi-agent smoke test.

Reads keys from env. We route the working model (nvidia/llama-3.3-nemotron-super-49b-v1)
since the nano-omni model in the original skill is currently degraded by NVIDIA.
"""
import os
import requests

NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")

# Working model (the nemotron-3-nano-omni-30b in the original skill is degraded).
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def llm(prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Call NVIDIA Nemotron chat completions. Returns assistant content string."""
    api_key = os.environ.get("NVIDIA_API_KEY") or NVIDIA_KEY
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is required.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        NVIDIA_URL,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json={"messages": messages, "model": NVIDIA_MODEL, "max_tokens": max_tokens,
              "stream": False, "temperature": temperature},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def search(query: str, max_results: int = 5):
    """Tavily web search. Returns the raw results list."""
    api_key = os.environ.get("TAVILY_API_KEY") or TAVILY_KEY
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required.")
    from tavily import TavilyClient
    return TavilyClient(api_key=api_key).search(query, max_results=max_results)
