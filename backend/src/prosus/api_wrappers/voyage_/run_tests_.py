from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels
from functools import cache
from dotenv import load_dotenv
from prosus.constants import test_query as query, test_documents as documents
load_dotenv()
import asyncio

@cache
def get_test_wrapper() -> VoyageAIModelAPI:
    return VoyageAIModelAPI(show_progress_bar=True)

async def test_wrapper__aembed_queries() -> None:
    wrapper = get_test_wrapper()
    vecs = await wrapper.aembed_queries(
        ["first query", "second query"]
    )
    print(len(vecs), len(vecs[0]))

async def test_wrapper__aembed_documents() -> None:
    wrapper = get_test_wrapper()
    fake_docs = [f"test document number{i}" for i in range(200)]
    vecs = await wrapper.aembed_documents(texts=fake_docs)
    print(len(vecs), len(vecs[0]))

async def test_wrapper__arerank_documents() -> None:
    """Test reranking functionality with a query and 20 diverse sentences."""
    # Initialize wrapper with a reranking model
    wrapper = VoyageAIModelAPI(
        model_name=VoyageAIModels.RERANK_2_5,
        show_progress_bar=True
    )
    
    print(f"\n{'='*80}")
    print(f"RERANKING TEST")
    print(f"{'='*80}")
    print(f"\nQuery: '{query}'")
    print(f"\nReranking {len(documents)} documents...\n")

    # Perform reranking
    results = await wrapper.arerank_documents(
        query=query,
        documents=documents
    )

    # Display results sorted by relevance
    print(f"{'Rank':<6} {'Score':<8} {'Document'}")
    print(f"{'-'*80}")

    for rank, result in enumerate(results, 1):
        score = result['relevance_score']
        original_index = result['index']

        doc = documents[original_index]
        # Truncate document if too long for display
        display_doc = doc if len(doc) <= 65 else doc[:62] + "..."

        print(f"{rank:<6} {score:<8.4f} {display_doc}")

    # Summary statistics
    print(f"\n{'-'*80}")
    print(f"Top score: {results[0]['relevance_score']:.4f}")
    print(f"Bottom score: {results[-1]['relevance_score']:.4f}")
    print(f"Score range: {results[0]['relevance_score'] - results[-1]['relevance_score']:.4f}")

    await wrapper.aclose_client()

if __name__ == "__main__":
    # asyncio.run(test_wrapper__aembed_queries())
    # asyncio.run(test_wrapper__aembed_documents())
    asyncio.run(test_wrapper__arerank_documents())
    print("\nAll tests passed!")