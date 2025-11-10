"""
FastAPI server for the semantic search API.
Provides an endpoint to query the matching pipeline and return top item IDs.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import os
import csv
import json

# Add the prosus package to the path so we can import match_query
sys.path.insert(0, str(Path(__file__).parent.parent))

from prosus.match_query import match_query, RerankStrategy, PipelineParameters
from prosus.search_indexes.orch import (
    build_and_return__bm25_combined_description_index,
    build_and_return__faiss_combined_description_index,
    build_and_return__faiss_tags_hooks_index,
    build_and_return__faiss_clip_image_index
)

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="API for semantic search over food items",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite default dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving images and data
# Get the path to the data directory (backend/data)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
IMAGES_DIR = DATA_DIR / "downloaded_images"
ITEMS_CSV = DATA_DIR / "5k_items_curated.csv"

# Mount the images directory so frontend can access images via /images/{filename}
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")
    print(f"Mounted images directory: {IMAGES_DIR}")
else:
    print(f"Warning: Images directory not found at {IMAGES_DIR}")

# Global variable to store items data (loaded on startup)
items_by_id: Dict[str, Dict[str, Any]] = {}


def load_items_data():
    """
    Load items data from CSV and create a lookup dictionary by itemId.
    """
    global items_by_id
    items_by_id = {}

    if not ITEMS_CSV.exists():
        print(f"Warning: Items CSV not found at {ITEMS_CSV}")
        return

    try:
        with open(ITEMS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get('itemId')
                if item_id:
                    # Parse the JSON fields
                    metadata = json.loads(row.get('itemMetadata', '{}'))

                    # Convert image paths from CSV format (UUID/filename) to actual format (UUID_filename)
                    csv_images = metadata.get('images', [])
                    converted_images = [img.replace('/', '_') for img in csv_images]

                    items_by_id[item_id] = {
                        'itemId': item_id,
                        'name': metadata.get('name', ''),
                        'description': metadata.get('description', ''),
                        'price': metadata.get('price', 0),
                        'images': converted_images,
                        'category_name': metadata.get('category_name', ''),
                        'taxonomy': metadata.get('taxonomy', {}),
                        'tags': metadata.get('tags', [])
                    }
        print(f"Loaded {len(items_by_id)} items from CSV")
    except Exception as e:
        print(f"Error loading items data: {e}")


@app.on_event("startup")
async def startup_event():
    """
    Build all search indexes on server startup.
    This ensures indexes are ready before any queries are received.
    """
    print("\n" + "="*80)
    print("INITIALIZING SEARCH INDEXES AND DATA...")
    print("="*80)

    try:
        # Load items data from CSV
        print("\n[0/4] Loading items data from CSV...")
        load_items_data()

        # Build BM25 index on combined descriptions
        print("\n[1/4] Building BM25 combined description index...")
        bm25_index, bm25_items = build_and_return__bm25_combined_description_index()
        print(f"✓ BM25 index ready with {len(bm25_items)} items")

        # Build FAISS index on combined description embeddings
        print("\n[2/4] Building FAISS combined description index...")
        faiss_desc_index, faiss_desc_items = build_and_return__faiss_combined_description_index()
        print(f"✓ FAISS description index ready with {len(faiss_desc_items)} items")

        # Build FAISS index on tags and hooks embeddings
        print("\n[3/4] Building FAISS tags/hooks index...")
        faiss_tags_index, faiss_tags_words = build_and_return__faiss_tags_hooks_index()
        print(f"✓ FAISS tags/hooks index ready with {len(faiss_tags_words)} tags/hooks")

        # Build FAISS CLIP image index
        print("\n[4/4] Building FAISS CLIP image index...")
        faiss_clip_index, clip_item_ids = build_and_return__faiss_clip_image_index()
        print(f"✓ FAISS CLIP image index ready with {len(clip_item_ids)} image embeddings")

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
    rerank_strategy: str = Field("replace", description="Reranking strategy: 'multiply' or 'replace'")


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
        search_query: SearchQuery object containing the query string, top_k parameter, and rerank_strategy

    Returns:
        SearchResponse: Object containing the top matching item IDs

    Raises:
        HTTPException: If the search fails or invalid rerank_strategy is provided
    """
    try:
        # Validate rerank_strategy parameter
        if search_query.rerank_strategy not in ["multiply", "replace"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rerank_strategy: {search_query.rerank_strategy}. Must be 'multiply' or 'replace'"
            )

        # Convert string to RerankStrategy enum
        strategy = RerankStrategy(search_query.rerank_strategy)

        # Create pipeline parameters with the specified rerank strategy
        params = PipelineParameters(rerank_strategy=strategy)

        # Run the matching pipeline with the specified parameters
        results = await match_query(search_query.query, params)

        # Extract item IDs from top_k results
        item_ids = [result["item_id"] for result in results[:search_query.top_k]]

        return SearchResponse(
            query=search_query.query,
            item_ids=item_ids,
            count=len(item_ids)
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return HTTP 500
        print(f"Error during search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/items/{item_id}")
async def get_item(item_id: str):
    """
    Get item details by item ID.

    Args:
        item_id: The item ID to look up

    Returns:
        Item details including name, description, price, images, etc.

    Raises:
        HTTPException: If item not found
    """
    if item_id not in items_by_id:
        raise HTTPException(
            status_code=404,
            detail=f"Item {item_id} not found"
        )

    return items_by_id[item_id]


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
