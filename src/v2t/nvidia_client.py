"""NVIDIA NIM API client with 3-key rotation for VLM calls."""
from __future__ import annotations
import base64
import os
import time
from pathlib import Path
from typing import Optional, Union

from openai import OpenAI
from PIL import Image

from src.v2t.config import CFG


def load_keys() -> list[str]:
    """Load valid NVIDIA API keys from config, filtering placeholders and empty values."""
    keys = []
    for k in CFG.nvidia_api_keys:
        if k and k != "nvapi-REPLACE_ME_1" and not k.startswith("nvapi-REPLACE_ME_"):
            keys.append(k)
    return keys


class NvidiaKeyRotator:
    """Rotates through multiple NVIDIA API keys for rate limit handling."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_index = 0
        self.total_keys = len(keys)
        self.rotation_count = 0
        self.last_rotation_time = 0.0
        self.key_usage: dict[str, int] = {k: 0 for k in keys}

    def get_active(self) -> str:
        """Get the currently active API key."""
        if not self.keys:
            raise ValueError("No valid NVIDIA API keys available")
        return self.keys[self.current_index]

    def rotate(self) -> str:
        """Rotate to the next available key."""
        if self.total_keys <= 1:
            return self.get_active()

        self.current_index = (self.current_index + 1) % self.total_keys
        self.rotation_count += 1
        self.last_rotation_time = time.time()
        return self.get_active()

    def record_usage(self, key: Optional[str] = None) -> None:
        """Record usage for a key (defaults to current active key)."""
        if key is None:
            key = self.get_active()
        if key in self.key_usage:
            self.key_usage[key] += 1

    def get_status(self) -> dict:
        """Get rotator status for debugging."""
        return {
            "total_keys": self.total_keys,
            "current_index": self.current_index,
            "current_key_prefix": self.get_active()[:20] + "..." if self.total_keys > 0 else "none",
            "rotation_count": self.rotation_count,
            "key_usage": self.key_usage,
        }


# Global rotator instance
_keys = load_keys()
_rotator = NvidiaKeyRotator(_keys) if _keys else None


def get_nvidia_vlm_client() -> OpenAI:
    """Get OpenAI client configured for NVIDIA NIM with active API key."""
    if _rotator is None:
        raise RuntimeError("No NVIDIA API keys configured. Set NVIDIA_API_KEY_1/2/3 in .env")

    api_key = _rotator.get_active()
    _rotator.record_usage(api_key)

    return OpenAI(
        base_url=CFG.nvidia_base_url,
        api_key=api_key,
    )


def image_to_base64(image: Union[str, Path, Image.Image]) -> str:
    """Convert image file path or PIL Image to base64 string."""
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def call_vlm(
    image: Union[str, Path, Image.Image],
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """
    Call NVIDIA NIM VLM with automatic key rotation on 429 errors.

    Args:
        image: Image path or PIL Image
        prompt: Text prompt for VLM
        model: Model name (defaults to CFG.nvidia_vlm_model)
        max_tokens: Max tokens in response
        temperature: Sampling temperature
        max_retries: Max retry attempts with key rotation

    Returns:
        Response text from VLM

    Raises:
        RuntimeError: If all retries exhausted or keys not configured
    """
    if _rotator is None or _rotator.total_keys == 0:
        raise RuntimeError("No NVIDIA API keys configured. Set NVIDIA_API_KEY_1/2/3 in .env")

    model = model or CFG.nvidia_vlm_model
    image_b64 = image_to_base64(image)

    last_error = None

    for attempt in range(max_retries):
        try:
            client = get_nvidia_vlm_client()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check for rate limit (429) or quota exceeded
            if "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), rotating key...")
                    _rotator.rotate()
                    time.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue

            # For other errors, don't retry with key rotation
            raise

    raise RuntimeError(f"VLM call failed after {max_retries} retries: {last_error}")


def is_keys_configured() -> bool:
    """Check if at least one valid NVIDIA API key is configured."""
    return _rotator is not None and _rotator.total_keys > 0


def run_stage_1() -> dict:
    """Test NVIDIA client configuration and key rotation."""
    print("=== Stage 1: NVIDIA Client Test ===")

    result = {
        "keys_configured": is_keys_configured(),
        "num_valid_keys": len(_keys),
        "keys": [k[:20] + "..." for k in _keys] if _keys else [],
    }

    if not is_keys_configured():
        print("WARNING: No valid NVIDIA API keys found")
        print("Set NVIDIA_API_KEY_1, NVIDIA_API_KEY_2, NVIDIA_API_KEY_3 in .env")
        result["status"] = "skipped_no_keys"
        return result

    # Test key rotator
    print(f"Loaded {_rotator.total_keys} valid keys")
    print(f"Active key: {_rotator.get_active()[:20]}...")
    print(f"Rotator status: {_rotator.get_status()}")

    # Test client creation
    try:
        client = get_nvidia_vlm_client()
        print("OpenAI client created successfully")
        result["client_created"] = True
    except Exception as e:
        print(f"Failed to create client: {e}")
        result["client_created"] = False
        result["error"] = str(e)

    result["status"] = "ok"
    return result


if __name__ == "__main__":
    import json
    result = run_stage_1()
    print(json.dumps(result, indent=2))