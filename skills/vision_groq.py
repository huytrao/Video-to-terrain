from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODELS = [
    "qwen/qwen3.6-27b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

IMAGE_URL = "https://example.com/image.jpg"

PROMPT = """
Extract all text from this image.

Rules:
- Return only the extracted text.
- Do not summarize.
- Do not translate.
- Preserve line breaks.
- Preserve numbers exactly.
- Preserve punctuation.
- If text is unclear, write [UNCLEAR].
"""

for model in MODELS:
    print("=" * 80)
    print(f"MODEL: {model}")
    print("=" * 80)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": IMAGE_URL
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_completion_tokens=4096,
        )

        print(response.choices[0].message.content)

    except Exception as e:
        print(f"ERROR: {e}")