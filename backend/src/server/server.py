"""
FastAPI server for the semantic search API.
Provides an endpoint to query the matching pipeline and return top item IDs.
"""

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
from typing import List
import sys
from pathlib import Path

# Add the prosus package to the path so we can import match_query
sys.path.insert(0, str(Path(__file__).parent.parent))

from prosus.match_query import match_query
from prosus.search_indexes.orch import (
    build_and_return__bm25_combined_description_index,
    build_and_return__faiss_combined_description_index,
    build_and_return__faiss_tags_hooks_index
)

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="API for semantic search over food items",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """
    Build all search indexes on server startup.
    This ensures indexes are ready before any queries are received.
    """
    print("\n" + "="*80)
    print("INITIALIZING SEARCH INDEXES...")
    print("="*80)

    try:
        # Build BM25 index on combined descriptions
        print("\n[1/3] Building BM25 combined description index...")
        bm25_index, bm25_items = build_and_return__bm25_combined_description_index()
        print(f"✓ BM25 index ready with {len(bm25_items)} items")

        # Build FAISS index on combined description embeddings
        print("\n[2/3] Building FAISS combined description index...")
        faiss_desc_index, faiss_desc_items = build_and_return__faiss_combined_description_index()
        print(f"✓ FAISS description index ready with {len(faiss_desc_items)} items")

        # Build FAISS index on tags and hooks embeddings
        print("\n[3/3] Building FAISS tags/hooks index...")
        faiss_tags_index, faiss_tags_words = build_and_return__faiss_tags_hooks_index()
        print(f"✓ FAISS tags/hooks index ready with {len(faiss_tags_words)} tags/hooks")

        print("\n" + "="*80)
        print("ALL INDEXES INITIALIZED SUCCESSFULLY!")
        print("Server is ready to handle search queries")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"ERROR: Failed to initialize indexes: {str(e)}")
        print("="*80 + "\n")
        raise

# Request model for search queries
class SearchQuery(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query string", min_length=1)
    top_k: int = Field(10, description="Number of top results to return", ge=1, le=100)


# Response model for search results
class SearchResponse(BaseModel):
    """Response model containing top item IDs."""
    query: str = Field(..., description="The original search query")
    item_ids: List[str] = Field(..., description="List of top matching item IDs")
    count: int = Field(..., description="Number of results returned")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Semantic Search API is running. Use POST /search to query."
    }


@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery):
    """
    Search for items matching the query.

    Args:
        search_query: SearchQuery object containing the query string and top_k parameter

    Returns:
        SearchResponse: Object containing the top matching item IDs

    Raises:
        HTTPException: If the search fails
    """
    try:
        # Run the matching pipeline
        results = await match_query(search_query.query)

        # Extract item IDs from top_k results
        item_ids = [result["item_id"] for result in results[:search_query.top_k]]

        return SearchResponse(
            query=search_query.query,
            item_ids=item_ids,
            count=len(item_ids)
        )

    except Exception as e:
        # Log the error and return HTTP 500
        print(f"Error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}

if __name__ == "__main__":

    # run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8080,
        reload=False
    )
