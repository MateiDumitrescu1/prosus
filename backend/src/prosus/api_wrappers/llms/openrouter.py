# OpenRouter API wrapper methods for LLM calls

import os
from functools import lru_cache
from dotenv import load_dotenv
from openai import AsyncOpenAI
from enum import StrEnum
import asyncio

open_router_providers = {
    "order": ["groq", "cerebras"],
    "sort": "throughput",
}

class OpenRouterModel(StrEnum):
    """Available OpenRouter models"""
    GPT_OSS_120B = "openai/gpt-oss-120b"
    KIMI_K2 = "moonshotai/kimi-k2-0905"

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

print(f"Using OpenRouter API key: {OPENROUTER_API_KEY[:10]}...")

@lru_cache(maxsize=1)
def get_openrouter_async_client() -> AsyncOpenAI:
    """
    Get a cached AsyncOpenAI client configured for OpenRouter.

    Returns:
        AsyncOpenAI client instance configured with OpenRouter base URL
    """
    openrouter_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    return openrouter_client

async def get_openrouter_llm_response(
    prompt: str,
    model: OpenRouterModel = OpenRouterModel.GPT_OSS_120B,
    system_message: str = "You are a helpful assistant.",
) -> str:
    """
    Send a prompt to OpenRouter and get a text response.

    Args:
        prompt: The user prompt to send to the model
        model: The OpenRouter model to use
        system_message: System message to set assistant behavior

    Returns:
        The model's response as a string
    """
    openrouter_client = get_openrouter_async_client()
    response = await openrouter_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        extra_body={"provider": open_router_providers}
    )
    return response.choices[0].message.content


#! ---- TESTING ----

async def test_get_openrouter_llm_response():
    """Test the OpenRouter LLM response function"""
    prompt = "What is the capital of France?"
    response = await get_openrouter_llm_response(prompt)
    print("OpenRouter LLM Response:", response)

if __name__ == "__main__":
    asyncio.run(test_get_openrouter_llm_response())
