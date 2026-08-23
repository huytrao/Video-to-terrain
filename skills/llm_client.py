"""Shared NVIDIA Nemotron + Tavily client helpers used by the multi-agent smoke test.

Reads keys from env if present, falls back to the keys bundled in skills/. We route
the working model (nvidia/llama-3.3-nemotron-super-49b-v1) since the nano-omni model
in the original skill is currently degraded by NVIDIA.
"""
import os
import requests

NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "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")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "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")

# Working model (the nemotron-3-nano-omni-30b in the original skill is degraded).
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def llm(prompt: str, system: str = "", max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Call NVIDIA Nemotron chat completions. Returns assistant content string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        NVIDIA_URL,
        headers={"Authorization": f"Bearer {NVIDIA_KEY}", "Accept": "application/json"},
        json={"messages": messages, "model": NVIDIA_MODEL, "max_tokens": max_tokens,
              "stream": False, "temperature": temperature},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def search(query: str, max_results: int = 5):
    """Tavily web search. Returns the raw results list."""
    from tavily import TavilyClient
    return TavilyClient(api_key=TAVILY_KEY).search(query, max_results=max_results)
