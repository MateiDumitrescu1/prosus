"""
Test script to demonstrate integration of FAISS index with Voyage AI embeddings.
Uses test data from constants.py to perform semantic search.
"""
import asyncio
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from prosus.constants import test_query, tesst_documents
from prosus.search_indexes.faiss_.faiss_ops import FaissIndex
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels

#! test method
async def test_faiss_with_voyage_embeddings():
    """
    Test FAISS index operations using Voyage AI embeddings.

    This test demonstrates:
    1. Embedding documents using Voyage AI (document mode)
    2. Creating and initializing a FAISS index
    3. Embedding a query using Voyage AI (query mode)
    4. Performing semantic search to find relevant documents
    """

    print("=" * 80)
    print("FAISS + Voyage AI Integration Test")
    print("=" * 80)

    #* Initialize the Voyage AI API wrapper
    print("\n[Step 1] Initializing Voyage AI API...")
    voyage_api = VoyageAIModelAPI(
        model_name=VoyageAIModels.VOYAGE_3_5,
        show_progress_bar=True
    )
    print(f"Using model: {voyage_api.model_name}")

    #* Load test documents and query from constants
    print(f"\n[Step 2] Loading test data...")
    print(f"Query: '{test_query}'")
    print(f"Number of documents: {len(tesst_documents)}")

    #* Embed documents using document mode
    print("\n[Step 3] Embedding documents (using document mode)...")
    document_embeddings = await voyage_api.aembed_documents(tesst_documents)
    print(f"Generated {len(document_embeddings)} document embeddings")
    print(f"Embedding dimension: {len(document_embeddings[0])}")

    #* Convert embeddings to numpy array for FAISS
    document_embeddings_array = np.array(document_embeddings, dtype=np.float32)
    print(f"Document embeddings shape: {document_embeddings_array.shape}")

    #* Initialize FAISS index with document embeddings
    print("\n[Step 4] Initializing FAISS index...")
    faiss_index = FaissIndex()
    faiss_index.initialize(
        embeddings=document_embeddings_array,
        save_on_disk=False,
        metric_type="ip"  # Inner product for cosine similarity
    )
    print(f"Index size: {faiss_index.get_index_size()} vectors")
    print(f"Embedding dimension: {faiss_index.get_embedding_dimension()}")

    #* Embed query using query mode
    print("\n[Step 5] Embedding query (using query mode)...")
    query_embeddings = await voyage_api.aembed_queries([test_query])
    query_embedding = np.array(query_embeddings[0], dtype=np.float32)
    print(f"Query embedding shape: {query_embedding.shape}")

    #* Perform semantic search
    print("\n[Step 6] Performing semantic search...")
    top_k = 5
    distances, indices = faiss_index.search(query_embedding, top_k=top_k)

    #* Display results
    print(f"\n{'=' * 80}")
    print(f"Top {top_k} Most Relevant Documents")
    print(f"{'=' * 80}")
    print(f"\nQuery: '{test_query}'\n")

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        print(f"Rank {rank}:")
        print(f"  Document Index: {idx}")
        print(f"  Similarity Score: {score:.4f}")
        print(f"  Text: '{tesst_documents[idx]}'")
        print()

    #* Close the Voyage AI client
    print("[Step 7] Cleaning up...")
    await voyage_api.aclose_client()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


def main():
    """Main entry point for the test script."""
    try:
        asyncio.run(test_faiss_with_voyage_embeddings())
    except Exception as e:
        print(f"\nL Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
