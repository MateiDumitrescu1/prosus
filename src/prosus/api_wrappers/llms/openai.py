# here will go the methods to call the GPT API and the embeding model APIs

import os
from functools import lru_cache
from dotenv import load_dotenv
from openai import AsyncOpenAI
from enum import StrEnum
import asyncio
from pathlib import Path
from typing import Union
from prosus.utils.image_utils import image_to_base64
#!

class OpenAIModel(StrEnum):
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_4O = "gpt-4o"  # Vision-capable model
    GPT_4O_MINI = "gpt-4o-mini"  # Vision-capable model

class OpenAIEmbeddingModel(StrEnum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

load_dotenv()
OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

print(f" using openai key: {OPENAI_API_KEY}")

@lru_cache(maxsize=1)
def get_openai_async_client() -> AsyncOpenAI:
    openai_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY, 
        # base_url="https://pd67dqn1bd.execute-api.eu-west-1.amazonaws.com"
    )
    return openai_client

async def get_openai_llm_response(
    prompt: str,
    model: OpenAIModel = OpenAIModel.GPT_4_1_MINI,
) -> str:
    print
    openai_client = get_openai_async_client()
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

async def get_openai_embedding(
    input_text: str,
    model: OpenAIEmbeddingModel = OpenAIEmbeddingModel.TEXT_EMBEDDING_3_SMALL,
) -> list[float]:
    openai_client = get_openai_async_client()
    response = await openai_client.embeddings.create(
        model=model,
        input=input_text,
    )
    return response.data[0].embedding


async def get_openai_vision_response(
    image_inputs: Union[list[Union[str, Path]], Union[str, Path]],
    text_prompt: str,
    model: OpenAIModel = OpenAIModel.GPT_5_NANO,
    system_message: str = "You are a helpful assistant that can analyze images.",
) -> str:
    """
    Send a text + image(s) prompt to OpenAI's Vision API.

    Args:
        image_inputs: Single image or list of images (file paths or base64 strings)
        text_prompt: Text prompt to accompany the images
        model: OpenAI model to use
        system_message: System message for the assistant

    Returns:
        The model's response as a string
    """

    #* Normalize to list for uniform processing
    if not isinstance(image_inputs, list):
        image_inputs = [image_inputs]

    #* Process all images to base64
    base64_images = []
    for image_input in image_inputs:
        # Determine if input is a file path or base64 string
        # Base64 strings with data URI start with "data:", otherwise treat as file path
        if isinstance(image_input, (str, Path)) and not str(image_input).startswith("data:"):
            # Check if it's a valid file path
            image_path = Path(image_input)
            if image_path.exists():
                # Convert file to base64 with data URI prefix
                base64_images.append(image_to_base64(str(image_input), include_data_uri=True))
            else:
                raise FileNotFoundError(f"Image file not found: {image_input}")
        else:
            # Already a base64 string with data URI prefix
            base64_images.append(str(image_input))

    #* Build content array: text first, then all images
    content = [{"type": "text", "text": text_prompt}]
    content.extend([
        {
            "type": "image_url",
            "image_url": {"url": img}
        }
        for img in base64_images
    ])

    #* Create the messages with vision content
    openai_client = get_openai_async_client()
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": content
            }
        ],
    )
    return response.choices[0].message.content

#! util methods derived from the LLM api call wrapper
async def translate_text(
    text: str,
    target_language: str
) -> str:
    prompt = f"Translate the following text to {target_language}: {text}"
    response = await get_openai_llm_response(prompt)
    return response

#! util methods derived from the LLM api call wrapper

#! ---- TESTING ----

async def test_get_openai_llm_response():
    prompt = "What is the capital of France?"
    response = await get_openai_llm_response(prompt)
    print("LLM Response:", response)
    
async def test_get_openai_embedding():
    input_text = "Hello, world!"
    embedding = await get_openai_embedding(input_text)
    print("Embedding:", embedding)

maggi_soup_image = "../../data/downloaded_images/820af392-002c-47b1-bfae-d7ef31743c7f_202402200931_gxgyfoywbcj.jpeg"
async def test_get_openai_vision_response():
    """
    Test the vision API with a sample image.
    Note: You'll need to provide a valid image path to test this.
    """
    try:
        # Example with a file path (update with actual image path)
        image_path = maggi_soup_image
        prompt = "Describe what you see in this image."
        response = await get_openai_vision_response(image_path, prompt)
        print(response)
    except FileNotFoundError:
        print("Test skipped: Please provide a valid image path in the test function")

if __name__ == "__main__":
    # asyncio.run(test_get_openai_llm_response())
    # asyncio.run(test_get_openai_embedding())
    asyncio.run(test_get_openai_vision_response()) 

    