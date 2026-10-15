import os
import requests

api_key = os.environ.get("NVIDIA_API_KEY")
if not api_key:
    raise ValueError("NVIDIA_API_KEY environment variable is required.")

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
  "max_tokens": 65536,
  "reasoning_budget": 16384,
  "stream": stream,
  "temperature": 0.6,
  "top_p": 0.95
}

response = requests.post(invoke_url, headers=headers, json=payload, stream=stream)
if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())