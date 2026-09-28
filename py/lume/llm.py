import os
import base64
from pathlib import Path
import httpx

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# `deepseek-flash` is the API identifier for DeepSeek-V4.1-Flash.
DEEPSEEK_MODEL = "deepseek-flash"


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set")
    return key


class Usage:
    def __init__(self, data: dict):
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)
        self.cache_hit = data.get("prompt_cache_hit_tokens", 0)
        self.cache_miss = data.get("prompt_cache_miss_tokens", 0)


class LLMResponse:
    def __init__(self, content: str, usage: Usage, model: str):
        self.content = content
        self.usage = usage
        self.model = model


async def call_deepseek(
    system_prompt: str, messages: list[dict], model: str = DEEPSEEK_MODEL
) -> LLMResponse:
    # Keep the public argument for backward compatibility with integrations,
    # but route every request through the single supported model.
    model = DEEPSEEK_MODEL
    api_key = get_api_key()

    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            DEEPSEEK_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=body,
        )
        if resp.is_error:
            raise RuntimeError(
                f"DeepSeek API request failed ({resp.status_code}): {resp.text}"
            )

    data = resp.json()
    usage = Usage(data.get("usage", {}))
    content = data["choices"][0]["message"]["content"]
    model = data.get("model", model)
    return LLMResponse(content=content, usage=usage, model=model)


def analyze_image(image_path: Path, prompt: str) -> str:
    """Ask the unified DeepSeek model to describe an image synchronously."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    body = {
        "model": DEEPSEEK_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
    }
    with httpx.Client(timeout=60) as client:
        response = client.post(
            DEEPSEEK_API_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {get_api_key()}"},
            json=body,
        )
    if response.is_error:
        raise RuntimeError(f"DeepSeek image request failed ({response.status_code}): {response.text}")
    return str(response.json()["choices"][0]["message"]["content"]).strip()
