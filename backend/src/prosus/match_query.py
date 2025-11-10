# this is the main pipeline
import json
from prosus.search_indexes.orch import (
    build_and_return__bm25_combined_description_index,
    build_and_return__faiss_combined_description_index,
    build_and_return__faiss_tags_hooks_index,
    build_and_return__faiss_clip_image_index,
)
from paths_ import matching_output_dir
from functools import cache
from pathlib import Path
import numpy as np
import asyncio
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels
from datetime import datetime
from enum import StrEnum
from sentence_transformers import SentenceTransformer
from prosus.constants import sentence_transformers_clip_multilingual_text__model_name

class RerankStrategy(StrEnum):
    MULTIPLY = "multiply" # the final score is: aggregated_score * relevance_score
    REPLACE = "replace" # the final score is simply the relevance_score

#! config
ground_truth_item_data = "../../data/5k_items_new_format@v1.jsonl" # we use this to get the tags and hooks for items
clip_image_index_score_multiplier = 0.75
bm25_tag_hooks_score_multiplier = 0.5

relevance_score_power = 1.0 # when using the MULTIPLY strategy, raise the relevance score to this power before multiplying. this helps to emphasize the reranker's relevance score more.
# set to 1.5 or 2.0 to penalize low relevance scores more heavily
#! config

@cache
def get_clip_text_model():
    """
    Load and cache the CLIP model for text encoding.
    Returns the SentenceTransformer CLIP model instance.
    """
    print(f"Loading CLIP model: {sentence_transformers_clip_multilingual_text__model_name}...")
    model = SentenceTransformer(sentence_transformers_clip_multilingual_text__model_name)
    print(f"CLIP model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model

@cache
def build_tag_hooks_lookup_dict():
    """
    Build a lookup dictionary for item_id to tags and hooks.
    tag/hook -> item_ids mapping.
    """
    word_to_id_mapping = {}

    with open(ground_truth_item_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            item_id = item['item_id']
            tags = item.get('tags', [])
            hooks = item.get('associated_keyword_hooks', [])
            all_words = tags + hooks

            for word in all_words:
                if word not in word_to_id_mapping:
                    word_to_id_mapping[word] = set()
                word_to_id_mapping[word].add(item_id)

    return word_to_id_mapping

@cache
def build_item_to_tags_hooks_count():
    """
    Build a lookup dictionary mapping item_id to the total count of tags and hooks.
    Returns dict: item_id -> count of tags/hooks
    """
    item_to_count = {}

    with open(ground_truth_item_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            item_id = item['item_id']
            tags = item.get('tags', [])
            hooks = item.get('associated_keyword_hooks', [])
            total_count = len(tags) + len(hooks)
            item_to_count[item_id] = total_count if total_count > 0 else 1  # Avoid division by zero

    return item_to_count

@cache
def build_item_descriptions_dict():
    """
    Build a lookup dictionary mapping item_id to a combined description.
    Returns dict: item_id -> combined description (name + description)
    """
    item_descriptions = {}

    with open(ground_truth_item_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            item_id = item['item_id']
            name = item.get('name', '')
            description = item.get('description', '')
            # Combine name and description for better context
            combined = f"{name}. {description}".strip()
            item_descriptions[item_id] = combined if combined else "No description available"

    return item_descriptions

common_top_k = 30 # all 4 search methods will return top 30 items

async def match_query(query:str, rerank_strategy: RerankStrategy = RerankStrategy.REPLACE):
    """
    Takes in a query and runs the whole pipeline to return relevant results.

    Args:
        query: The search query string
        rerank_strategy: Strategy for combining scores (MULTIPLY or REPLACE). Defaults to REPLACE.

    Pipeline:
    1. Search four different indexes and get 0-1 normalized scores for top items from each:
       - BM25 combined description index (text-based keyword matching)
       - FAISS combined description index (semantic similarity on descriptions)
       - FAISS tags/hooks index (semantic matching on tags and hooks)
       - FAISS CLIP image index (visual-semantic matching on product images)
    2. Apply score multipliers to specific search methods (clip_image_index_score_multiplier, bm25_tag_hooks_score_multiplier)
    3. Aggregate scores across all search methods and create a shortlist
    4. Use Voyage AI reranker to rerank the shortlisted items
    5. Apply the selected reranking strategy to compute final scores
    6. Modify the scores based on total_orders and reorder_rate (popularity boost - future enhancement)
    7. Return the final ranked list of items with scores
    """

    # Compute query embeddings once at the top (used by FAISS indexes)
    voyage_api = VoyageAIModelAPI()
    query_embedding = await voyage_api.aembed_queries([query])
    query_embedding_array = np.array(query_embedding[0], dtype=np.float32)

    # Generate CLIP query embedding for image search
    clip_model = get_clip_text_model()
    clip_query_embedding = clip_model.encode(query, convert_to_tensor=False, normalize_embeddings=True)
    clip_query_embedding_array = np.array([clip_query_embedding], dtype=np.float32)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = Path(matching_output_dir) / f"matching_results_{current_time}"

    # Create the output folder if it doesn't exist
    results_folder.mkdir(parents=True, exist_ok=True)
    
    async def search_in_bm25_combined_description_index(query:str):
        """
        Search in the BM25 combined description index.
        Normalize scores to 0-1 range.
        Returns list of dicts with item_id and score.
        """
        bm25_index, item_ids = build_and_return__bm25_combined_description_index()
        # BM25 search returns (results, scores) where results contains indices
        indices, scores = bm25_index.search(query, top_k=common_top_k, return_documents=False)

        # Normalize scores to 0-1 range (min-max normalization)
        scores_array = scores[0]  # Get first row (single query)
        indices_array = indices[0]  # Get first row (single query)

        if len(scores_array) > 0:
            max_score = np.max(scores_array)
            min_score = np.min(scores_array)
            # Avoid division by zero
            if max_score > min_score:
                normalized_scores = (scores_array - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores_array)
        else:
            normalized_scores = scores_array

        results = []
        for idx, score in zip(indices_array, normalized_scores):
            item_id = item_ids[idx]
            results.append({"item_id": item_id, "score": float(score)})
            
        return results
    
    async def search_in_faiss_combined_description_index(query_embedding_array):
        """
        Search in the FAISS combined description index.
        Returns list of dicts with item_id and the (similarity) score.
        Scores are already in 0-1 range (cosine similarity from inner product).
        """
        faiss_index, item_ids = build_and_return__faiss_combined_description_index()

        # Search the FAISS index using precomputed query embedding
        distances, indices = faiss_index.search(query_embedding_array, top_k=common_top_k)

        # Format results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            item_id = item_ids[idx]
            # Distance is cosine similarity (inner product), already normalized
            results.append({"item_id": item_id, "score": float(distance)})
            
        return results

    async def search_in_faiss_tags_hooks_index(query_embedding_array):
        """
        Get the relevant tags/hooks by searching the FAISS tags/hooks index.
        Consider the top 50 results. Use  the word --> id mapping and assign scores to items whose tags/hooks are in the top results.
        The score for each item is: ( number of tags/hooks of that item present in top results ) / ( total number of tags/hooks for that item ).
        This is so the scores are in 0-1 range.
        """
        faiss_index, words = build_and_return__faiss_tags_hooks_index()
        word_to_id_mapping = build_tag_hooks_lookup_dict()
        item_to_count = build_item_to_tags_hooks_count()

        async def embed_query_and_search():
            # Search for top 50 similar tags/hooks using precomputed query embedding
            distances, indices = faiss_index.search(query_embedding_array, top_k=common_top_k)

            # Get the matched words (tags/hooks)
            matched_words = set()
            for idx in indices[0]:
                matched_words.add(words[idx])

            print(f"we have {len(matched_words)} matched tags/hooks for the query '{query}'")
            
            # Count how many matched tags/hooks each item has
            item_match_count = {}
            for word in matched_words:
                if word in word_to_id_mapping:
                    for item_id in word_to_id_mapping[word]:
                        item_match_count[item_id] = item_match_count.get(item_id, 0) + 1

            # Calculate normalized scores for each item
            results = []
            for item_id, match_count in item_match_count.items():
                total_count = item_to_count.get(item_id, 1)
                score = match_count / total_count
                results.append({"item_id": item_id, "score": score})

            # Sort by score descending and take top 10
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:10]

            return results

        return await embed_query_and_search()

    async def search_in_faiss_clip_image_index(clip_query_embedding_array):
        """
        Search in the FAISS CLIP image index.
        Returns list of dicts with item_id and the (similarity) score.
        The scores are clustered in a lower range (e.g., 0.25-0.35).
        Normalize them to 0-1 range using min-max normalization.
        """
        faiss_index, item_ids = build_and_return__faiss_clip_image_index()

        # Search the FAISS CLIP index using precomputed CLIP query embedding
        distances, indices = faiss_index.search(clip_query_embedding_array, top_k=common_top_k)

        # Extract scores and normalize to 0-1 range
        scores_array = distances[0]  # Cosine similarity scores
        indices_array = indices[0]

        if len(scores_array) > 0:
            max_score = np.max(scores_array)
            min_score = np.min(scores_array)
            # Avoid division by zero
            if max_score > min_score:
                normalized_scores = (scores_array - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores_array)
        else:
            normalized_scores = scores_array

        # Format results
        results = []
        for idx, score in zip(indices_array, normalized_scores):
            item_id = item_ids[idx]
            results.append({"item_id": item_id, "score": float(score)})

        return results

    #* Step 1: Search all four indexes and get normalized scores
    print("Searching BM25 combined description index...")
    bm25_results = await search_in_bm25_combined_description_index(query)
    print(f"\033[91mFinished BM25 search for query: '{query}'\033[0m")

    #~ Save BM25 intermediate results
    bm25_file = results_folder / "bm25_results.jsonl"
    with open(bm25_file, 'w', encoding='utf-8') as f:
        for result in bm25_results:
            f.write(json.dumps(result) + '\n')
    print(f"Saved BM25 intermediate results to {bm25_file}")

    print("Searching FAISS combined description index...")
    faiss_desc_results = await search_in_faiss_combined_description_index(query_embedding_array)
    print(f"\033[91mFinished FAISS combined description search for query: '{query}'\033[0m")

    #~ Save FAISS description intermediate results
    faiss_desc_file = results_folder / "faiss_description_results.jsonl"
    with open(faiss_desc_file, 'w', encoding='utf-8') as f:
        for result in faiss_desc_results:
            f.write(json.dumps(result) + '\n')
    print(f"Saved FAISS description intermediate results to {faiss_desc_file}")

    print("Searching FAISS tags/hooks index...")
    faiss_tags_results = await search_in_faiss_tags_hooks_index(query_embedding_array)
    print(f"\033[91mFinished FAISS tags/hooks search for query: '{query}'\033[0m")

    #~ Save FAISS tags/hooks intermediate results
    faiss_tags_file = results_folder / "faiss_tags_hooks_results.jsonl"
    with open(faiss_tags_file, 'w', encoding='utf-8') as f:
        for result in faiss_tags_results:
            f.write(json.dumps(result) + '\n')
    print(f"Saved FAISS tags/hooks intermediate results to {faiss_tags_file}")

    print("Searching FAISS CLIP image index...")
    clip_image_results = await search_in_faiss_clip_image_index(clip_query_embedding_array)
    print(f"\033[91mFinished FAISS CLIP image search for query: '{query}'\033[0m")

    #~ Save FAISS CLIP image intermediate results
    clip_image_file = results_folder / "clip_image_results.jsonl"
    with open(clip_image_file, 'w', encoding='utf-8') as f:
        for result in clip_image_results:
            f.write(json.dumps(result) + '\n')
    print(f"Saved FAISS CLIP image intermediate results to {clip_image_file}")

    #* Step 2: Apply score multipliers and aggregate scores by item_id
    aggregated_scores = {}

    # Add BM25 scores (no multiplier for BM25 description)
    for result in bm25_results:
        item_id = result["item_id"]
        aggregated_scores[item_id] = aggregated_scores.get(item_id, 0) + result["score"]

    # Add FAISS description scores (no multiplier)
    for result in faiss_desc_results:
        item_id = result["item_id"]
        aggregated_scores[item_id] = aggregated_scores.get(item_id, 0) + result["score"]

    # Add FAISS tags/hooks scores with multiplier
    for result in faiss_tags_results:
        item_id = result["item_id"]
        weighted_score = result["score"] * bm25_tag_hooks_score_multiplier
        aggregated_scores[item_id] = aggregated_scores.get(item_id, 0) + weighted_score

    # Add CLIP image scores with multiplier
    for result in clip_image_results:
        item_id = result["item_id"]
        weighted_score = result["score"] * clip_image_index_score_multiplier
        aggregated_scores[item_id] = aggregated_scores.get(item_id, 0) + weighted_score

    #* Create shortlist of top items (sort by aggregated score)
    shortlist = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)

    # Take top candidates for reranking (e.g., top 50 or all if fewer)
    max_rerank_candidates = 50
    shortlist = shortlist[:max_rerank_candidates]

    if len(shortlist) == 0:
        print("No results found")
        return []

    print(f"Shortlisted {len(shortlist)} items for reranking")

    #* Step 3: Use Voyage AI reranker to rerank the shortlisted items
    item_descriptions = build_item_descriptions_dict()

    # Prepare documents for reranking
    rerank_item_ids = [item_id for item_id, _ in shortlist]
    rerank_documents = [item_descriptions.get(item_id, "") for item_id in rerank_item_ids]

    # Call the reranker
    print(f"Reranking with Voyage AI (strategy: {rerank_strategy})...")
    voyage_api = VoyageAIModelAPI(model_name=VoyageAIModels.RERANK_2_5)

    reranked_results = await voyage_api.arerank_documents(
        query=query,
        documents=rerank_documents,
        top_k=None  # Return all reranked results
    )

    #* Use rerank relevance scores as final scores based on selected strategy
    final_scores = []
    for rerank_result in reranked_results:
        original_index = rerank_result["index"]
        item_id = rerank_item_ids[original_index]
        aggregated_score = aggregated_scores[item_id]
        relevance_score = rerank_result["relevance_score"]

        # Apply reranking strategy
        if rerank_strategy == RerankStrategy.MULTIPLY:
            # Multiply aggregated score with relevance score raised to a power
            final_score = aggregated_score * (relevance_score ** relevance_score_power)
        elif rerank_strategy == RerankStrategy.REPLACE:
            # Use reranker's relevance score as the final score
            final_score = relevance_score
        else:
            raise ValueError(f"Unknown rerank strategy: {rerank_strategy}")

        final_scores.append({
            "item_id": item_id,
            "score": final_score,
            "aggregated_score": aggregated_score,
            "relevance_score": relevance_score
        })

    #* Step 4: Apply popularity boost (implement later)
    
    #* Step 5: Sort by final score and return
    final_scores.sort(key=lambda x: x["score"], reverse=True)

    #~ Save final results to JSONL file
    final_results_file = results_folder / "final_results.jsonl"
    with open(final_results_file, 'w', encoding='utf-8') as f:
        for result in final_scores:
            f.write(json.dumps(result) + '\n')
    print(f"Saved final results to {final_results_file}")

    print(f"Returning {len(final_scores)} results")
    return final_scores

#! -------- TESTING --------

async def test_match_query():
    query = "burger"
    results = await match_query(query)
    print(f"\nFinal Results for query: '{query}'")
    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}: Item ID: {result['item_id']}, Final Score: {result['score']:.4f}, "
              f"Aggregated Score: {result['aggregated_score']:.4f}, Relevance Score: {result['relevance_score']:.4f}")
        
if __name__ == "__main__":
    asyncio.run(test_match_query())