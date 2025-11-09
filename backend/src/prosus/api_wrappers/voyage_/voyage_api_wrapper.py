print("Initializing file: aembed.py")
import asyncio
import os
from typing import List, Optional, Literal, Iterable
import httpx
from functools import cache
from enum import StrEnum
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()
DEFAULT_CLIENT_TIMEOUT: float = 20.0
MAX_ALLOWED_BATCH_SIZE = 1000

#the input_type in the API call can also be None
class InputType(StrEnum):
    QUERY = "query"
    DOCUMENT = "document"

class VoyageAIModels(StrEnum):
    VOYAGE_3 = "voyage-3"
    VOYAGE_3_5_LITE = "voyage-3.5-lite"
    VOYAGE_3_5 = "voyage-3.5"
    RERANK_2 = "rerank-2"
    RERANK_2_LITE = "rerank-2-lite"
    RERANK_2_5 = "rerank-2.5"
    RERANK_2_5_LITE = "rerank-2.5-lite"

default_voyage_ai_embedding_model = VoyageAIModels.VOYAGE_3_5_LITE

@cache
def get_voyage_ai_API_KEY() -> str:
    key = os.getenv("VOYAGE_AI_API_KEY")
    if key is None: key = os.getenv("VOYAGE_API_KEY") # support both namings for the env variable
    
    if key is None:
        raise ValueError("get_voyage_ai_API_KEY method: VOYAGE_AI_API_KEY environment variable not set")
    
    return key

@cache
def get_voyage_ai_client(api_key: str | None = None, base_url: str = "https://api.voyageai.com/v1") -> httpx.AsyncClient:
    
    if api_key is None:
        api_key = get_voyage_ai_API_KEY()

    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=DEFAULT_CLIENT_TIMEOUT
    )

async def aembed_texts(
    texts: List[str],
    *,
    model: VoyageAIModels = VoyageAIModels.VOYAGE_3_5,
    input_type: InputType | None,
    #
    output_dimension: Optional[int] = None,    # e.g. 256, 512, 1024, 2048 (on supported models)
    output_dtype: str = "float",               # float | int8 | uint8 | binary | ubinary (on supported models)
    truncation: bool = True, #https://docs.voyageai.com/docs/embeddings Ctrl+F "over-length input texts will be truncated"
    #
    voyage_ai_client: Optional[httpx.AsyncClient] = get_voyage_ai_client(),
) -> List[List[float]]:
    """
    Minimal async wrapper for Voyage text embeddings.
    Returns a list of vectors, one per input string.

    Docs:
      - Endpoint & payload fields: /v1/embeddings (REST)  [docs]
      - Parameters (input_type, output_dimension/dtype, truncation, limits)  [guide]
      - Auth via VOYAGE_API_KEY                            [auth]
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be List[str]")

    if len(texts) > MAX_ALLOWED_BATCH_SIZE:
        raise ValueError(f"Max {MAX_ALLOWED_BATCH_SIZE} texts per request per Voyage docs.")

    payload = {"input": texts, "model": model, "truncation": truncation}
    if input_type is not None:
        payload["input_type"] = input_type
    if output_dimension is not None:
        payload["output_dimension"] = output_dimension
    if output_dtype is not None:
        payload["output_dtype"] = output_dtype

    r = await voyage_ai_client.post("/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()
    return [item["embedding"] for item in data["data"]]

async def arerank(
    query: str,
    documents: List[str],
    *,
    model: VoyageAIModels = VoyageAIModels.RERANK_2_5,
    top_k: Optional[int] = None,  # Return only top_k most relevant documents
    truncation: bool = True,      # Auto-truncate inputs exceeding context limits
    #
    voyage_ai_client: Optional[httpx.AsyncClient] = get_voyage_ai_client(),
) -> List[dict]:
    """
    Minimal async wrapper for Voyage reranking.
    Returns a list of reranked documents with relevance scores.

    Args:
        query: Search query (max 8,000 tokens for rerank-2.5/lite models)
        documents: List of up to 1,000 documents to rerank
        model: Reranking model to use
        top_k: Optional number of most relevant results to return
        truncation: Auto-truncate inputs exceeding context limits
        voyage_ai_client: HTTP client with API key configured

    Returns:
        List of dicts with keys:
            - index: Original position in documents list
            - document: Document text
            - relevance_score: Relevance score (0-1 range)
        Sorted by descending relevance.

    Docs:
      - Endpoint: POST /v1/rerank
      - Max documents: 1,000
      - Context limit: 32,000 tokens (query + any single document)
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
        raise TypeError("documents must be List[str]")

    if len(documents) > MAX_ALLOWED_BATCH_SIZE:
        raise ValueError(f"Max {MAX_ALLOWED_BATCH_SIZE} documents per request per Voyage docs.")

    payload = {
        "query": query,
        "documents": documents,
        "model": model,
        "truncation": truncation
    }

    if top_k is not None:
        payload["top_k"] = top_k

    r = await voyage_ai_client.post("/rerank", json=payload)
    r.raise_for_status()
    data = r.json()
    # print(data)
    # Return the reranked results with index, document text, and relevance score
    return data["data"]

class VoyageAIModelAPI:
    def __init__(self, 
            model_name: VoyageAIModels = default_voyage_ai_embedding_model, 
            show_progress_bar: bool = False
        ):
        self.api_key = get_voyage_ai_API_KEY()
        self.base_url = "https://api.voyageai.com/v1"
        self.async_client = get_voyage_ai_client(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.batch_size = MAX_ALLOWED_BATCH_SIZE  # default batch size for embedding requests
        self.show_progress_bar = show_progress_bar

    def _get_batch_iterator(self, texts: List[str]) -> Iterable:
        #! taken straight from the Langchain Wrapper library
        if self.show_progress_bar:
            _iter = tqdm(range(0, len(texts), self.batch_size))
        else:
            _iter = range(0, len(texts), self.batch_size)
        return _iter
    
    async def aembed_texts(self, texts: List[str], input_type: InputType | None) -> List[List[float]]:
        embeddings: List[List[float]] = []
        _iter = self._get_batch_iterator(texts=texts)
        
        for i in _iter:
            batch = texts[i:i + self.batch_size]
            this_batch_embeddings = await aembed_texts(
                voyage_ai_client=self.async_client,
                texts=batch,
                model=self.model_name,
                input_type=input_type
            )
            embeddings.extend(this_batch_embeddings)
            
        return embeddings
        

    async def aembed_queries(self, texts: List[str]) -> List[List[float]]:
        return await self.aembed_texts(
            texts=texts,
            input_type=InputType.QUERY
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.aembed_texts(
            texts=texts,
            input_type=InputType.DOCUMENT
        )

    async def arerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[dict]:
        """
        Rerank documents based on their relevance to a query.

        Args:
            query: Search query to rank documents against
            documents: List of document texts to rerank (max 1,000)
            top_k: Optional number of top results to return

        Returns:
            List of dicts sorted by relevance (descending), each containing:
                - index: Original position in documents list
                - document: Document text
                - relevance_score: Score from 0-1 indicating relevance
        """
        return await arerank(
            query=query,
            documents=documents,
            model=self.model_name,
            top_k=top_k,
            voyage_ai_client=self.async_client
        )

    async def aclose_client(self):
        await self.async_client.aclose()  # <-- let callers cleanly shut down
