import torch
from functools import cache
import json
import numpy as np
import asyncio
import os
import pandas as pd
from pathlib import Path
from paths_ import embeddings_output_dir, tags_and_hooks_embeddings_dir, combined_descriptions_dir, clip_image_embeddings_dir
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels, default_voyage_ai_embedding_model
from prosus.search_indexes.faiss_.faiss_ops import FaissIndex
from prosus.search_indexes.bm25_.bm25_ import BM25Index
from prosus.scripts.script_utils.combined_description_utils  import get_combined_descriptions_from_folder
from prosus.constants import sentence_transformers_clip_model_name

#! ---- config ----
version_to_use = "v0"
# combined_description_emb_file = "../../../data/embeddings_output/combined_description_embeddings/em_full_voyage-3.5-lite_20251109_182758/embeddings.jsonl"
combined_description_emb_file = Path(embeddings_output_dir) / "combined_description_embeddings" / "em_full_voyage-3.5-lite_20251109_182758" / "embeddings.jsonl"

# tags_and_hooks_emb_file = "../../../data/embeddings_output/tags_and_hooks_embeddings/tag_embeddings_voyage-3.5-lite_20251109_183615/embeddings.jsonl"
tags_and_hooks_emb_file = Path(embeddings_output_dir) / "tags_and_hooks_embeddings" / "tag_embeddings_voyage-3.5-lite_20251109_183615" / "embeddings.jsonl"

clip_emb_file = Path(clip_image_embeddings_dir) / "em_full_clip-ViT-B-32_20251110_025755" / "embeddings.jsonl"

csv_ground_truth_path = "../../../data/5k_items_curated.csv"
#! ---- config ----

#! BM25 on combined descriptions
@cache
def build_and_return__bm25_combined_description_index():
    """
    Build BM25 search index on combined descriptions.

    Returns:
        tuple: (BM25Index, item_ids) where:
            - BM25Index: Initialized BM25 index wrapper for keyword search
            - item_ids: List of item IDs corresponding to each document in the corpus
    """
    # Read combined descriptions from folder
    combined_descriptions = get_combined_descriptions_from_folder(version=version_to_use)

    # Prepare corpus and item IDs in consistent order
    item_ids = []
    corpus = []

    for item_id, description in combined_descriptions.items():
        item_ids.append(item_id)
        corpus.append(description)

    # Create and initialize the BM25 index
    bm25_index = BM25Index()
    bm25_index.initialize(
        corpus=corpus,
        stopwords="pt"  # Use Portuguese stopwords
    )

    print(f"Successfully built BM25 index with {len(item_ids)} combined descriptions")

    return bm25_index, item_ids

#! FAISS on combined descriptions
@cache
def build_and_return__faiss_combined_description_index():
    """
    Build FAISS search index on combined description embeddings.

    Returns:
        tuple: (FaissIndex, item_ids) where:
            - FaissIndex: Initialized FAISS index wrapper for similarity search
            - item_ids: List of item IDs corresponding to each embedding index
    """
    # Read embeddings from the JSONL file
    item_ids = []
    embeddings = []

    with open(combined_description_emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            item_ids.append(data['itemId'])
            embeddings.append(data['embedding'])

    # Convert embeddings to numpy array (float32 required by FAISS)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Create and initialize the FAISS index
    faiss_index = FaissIndex()
    faiss_index.initialize(
        embeddings=embeddings_array,
        save_on_disk=False,
        metric_type="ip"  # inner product for cosine similarity
    )

    print(f"Successfully built FAISS index with {len(item_ids)} combined description embeddings")

    return faiss_index, item_ids

#! FAISS on tags and hooks


@cache
def build_and_return__faiss_tags_hooks_index():
    """
    Build and return a FAISS index for tags and hooks embeddings.

    Returns:
        tuple: (FaissIndex, word_list) where:
            - FaissIndex: Initialized FAISS index wrapper for similarity search
            - word_list: List of words corresponding to each embedding index
    """

    # Read embeddings from the JSONL file
    words = []
    embeddings = []

    with open(tags_and_hooks_emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            words.append(data['word'])
            embeddings.append(data['embedding'])

    # Convert embeddings to numpy array (float32 required by FAISS)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Create and initialize the FAISS index
    faiss_index = FaissIndex()
    faiss_index.initialize(
        embeddings=embeddings_array,
        save_on_disk=False,
        metric_type="ip"  # inner product for cosine similarity
    )

    print(f"Successfully built FAISS index with {len(words)} tag/hook embeddings")

    return faiss_index, words

@cache
def build_and_return__faiss_clip_image_index():
    """
    Build FAISS search index on CLIP image embeddings.

    Returns:
        tuple: (FaissIndex, item_ids) where:
            - FaissIndex: Initialized FAISS index wrapper for similarity search
            - item_ids: List of item IDs corresponding to each embedding index
    """
    # Read embeddings from the JSONL file
    item_ids = []
    embeddings = []

    with open(clip_emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            item_ids.append(data['itemId'])
            embeddings.append(data['embedding'])

    # Convert embeddings to numpy array (float32 required by FAISS)
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Normalize embeddings for proper cosine similarity (L2 normalization)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / norms

    # Sanity check: pick a random embedding and ensure its norm is ~1
    if embeddings_array.shape[0] == 0:
        raise ValueError("No embeddings found in CLIP embeddings file")

    rand_idx = np.random.randint(0, embeddings_array.shape[0])
    vec = embeddings_array[rand_idx]
    norm = np.linalg.norm(vec)

    # Very close to 1 (allow tiny numerical tolerance)
    if not np.isclose(norm, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError(f"CLIP embedding at index {rand_idx} has norm {norm:.8f}, expected ~1.0")
    
    # Create and initialize the FAISS index
    faiss_index = FaissIndex()
    faiss_index.initialize(
        embeddings=embeddings_array,
        save_on_disk=False,
        metric_type="ip"  # inner product for cosine similarity
    )

    print(f"Successfully built FAISS CLIP image index with {len(item_ids)} embeddings")

    return faiss_index, item_ids

#! ---------------- TESTING ----------------
def test_bm25_combined_description_index():
    """
    Test the BM25 combined description index by searching for items matching 'burger'.
    """

    # Read the ground truth CSV and create a mapping of item IDs -> (name, description)
    print(f"Reading ground truth data from: {csv_ground_truth_path}")
    df = pd.read_csv(csv_ground_truth_path)

    item_metadata_map = {}
    for _, row in df.iterrows():
        item_id = row['itemId']
        # Parse the itemMetadata JSON string
        metadata = json.loads(row['itemMetadata'])
        name = metadata.get('name', 'N/A')
        description = metadata.get('description', 'N/A')
        item_metadata_map[item_id] = (name, description)

    print(f"Loaded metadata for {len(item_metadata_map)} items")

    # Build the BM25 index and get the item IDs
    bm25_index, item_ids = build_and_return__bm25_combined_description_index()
    print(f"\nBM25 combined description index: {bm25_index}")
    print(f"Total items indexed: {len(item_ids)}")

    test_query = "pizza"
    print(f"\nTest query: '{test_query}'")

    top_k = 10

    # Query the BM25 index for top results
    # The search method returns (results, scores) where results contains document indices or text
    results, scores = bm25_index.search(test_query, top_k=top_k, return_documents=False)

    print(f"\nTop {top_k} most similar items to '{test_query}':")
    print("=" * 100)
    for rank, (idx, score) in enumerate(zip(results[0], scores[0]), 1):
        item_id = item_ids[idx]
        name, description = item_metadata_map.get(item_id, ("Unknown", "N/A"))

        # Format and display the result with name and description
        print(f"{rank}. [BM25 Score: {score:.4f}]")
        print(f"   Item ID: {item_id}")
        print(f"   Name: {name}")
        print(f"   Description: {description}")
        print("-" * 100)
        
    print("=" * 100)

def test_faiss_tags_hooks_index():
    """
    Test the FAISS tags/hooks index by searching for similar tags.
    """
    # Build the FAISS index and get the word list
    faiss_index, words = build_and_return__faiss_tags_hooks_index()
    print(f"\nFAISS tags/hooks index: {faiss_index}")
    print(f"Total words indexed: {len(words)}")

    test_query = "burger"
    print(f"\nTest query: '{test_query}'")

    # Embed the query using the Voyage AI wrapper with the default model
    api_wrapper = VoyageAIModelAPI(
        model_name=default_voyage_ai_embedding_model,
        show_progress_bar=False
    )

    # Run the async embedding function
    async def embed_query():
        query_embedding = await api_wrapper.aembed_queries([test_query])
        return query_embedding[0]  # Return the first (and only) embedding

    # Get the query embedding
    query_embedding = asyncio.run(embed_query())
    query_vector = np.array(query_embedding, dtype=np.float32)

    top_k = 15
    
    # Query the index for top results
    distances, indices = faiss_index.search(query_vector, top_k=top_k)

    print(f"\nTop {top_k} most similar tags/hooks to '{test_query}':")
    print("-" * 60)
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        print(f"{rank}. '{words[idx]}' (similarity score: {distance:.4f})")
    print("-" * 60)

def test_faiss_combined_description_index():
    """
    Test the FAISS combined description index by searching for similar items.
    """

    # Read the ground truth CSV and create a mapping of item IDs -> (name, description)
    print(f"Reading ground truth data from: {csv_ground_truth_path}")
    df = pd.read_csv(csv_ground_truth_path)

    item_metadata_map = {}
    for _, row in df.iterrows():
        item_id = row['itemId']
        # Parse the itemMetadata JSON string
        metadata = json.loads(row['itemMetadata'])
        name = metadata.get('name', 'N/A')
        description = metadata.get('description', 'N/A')
        item_metadata_map[item_id] = (name, description)

    print(f"Loaded metadata for {len(item_metadata_map)} items")

    # Build the FAISS index and get the item IDs
    faiss_index, item_ids = build_and_return__faiss_combined_description_index()
    print(f"\nFAISS combined description index: {faiss_index}")
    print(f"Total items indexed: {len(item_ids)}")

    test_query = "burger"
    print(f"\nTest query: '{test_query}'")

    # Embed the query using the Voyage AI wrapper with the default model
    api_wrapper = VoyageAIModelAPI(
        model_name=default_voyage_ai_embedding_model,
        show_progress_bar=False
    )

    # Run the async embedding function
    async def embed_query():
        query_embedding = await api_wrapper.aembed_queries([test_query])
        return query_embedding[0]  # Return the first (and only) embedding

    # Get the query embedding
    query_embedding = asyncio.run(embed_query())
    query_vector = np.array(query_embedding, dtype=np.float32)

    top_k = 10

    # Query the index for top results
    distances, indices = faiss_index.search(query_vector, top_k=top_k)

    print(f"\nTop {top_k} most similar items to '{test_query}':")
    print("=" * 100)
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        item_id = item_ids[idx]
        name, description = item_metadata_map.get(item_id, ("Unknown", "N/A"))

        # Format and display the result with name and description
        print(f"{rank}. [Score: {distance:.4f}]")
        print(f"   Item ID: {item_id}")
        print(f"   Name: {name}")
        print(f"   Description: {description}")
        print("-" * 100)
    print("=" * 100)

def test_faiss_clip_image_index():
    """
    Test the FAISS CLIP image index by searching for similar items using a text query.
    The query is embedded using the same CLIP model as the images.
    """
    from sentence_transformers import SentenceTransformer

    # Read the ground truth CSV and create a mapping of item IDs -> (name, description)
    print(f"Reading ground truth data from: {csv_ground_truth_path}")
    df = pd.read_csv(csv_ground_truth_path)

    item_metadata_map = {}
    for _, row in df.iterrows():
        item_id = row['itemId']
        # Parse the itemMetadata JSON string
        metadata = json.loads(row['itemMetadata'])
        name = metadata.get('name', 'N/A')
        description = metadata.get('description', 'N/A')
        item_metadata_map[item_id] = (name, description)

    print(f"Loaded metadata for {len(item_metadata_map)} items")

    # Build the FAISS CLIP index and get the item IDs
    faiss_index, item_ids = build_and_return__faiss_clip_image_index()
    print(f"\nFAISS CLIP image index: {faiss_index}")
    print(f"Total items indexed: {len(item_ids)}")

    # Load the same CLIP model used for creating the image embeddings
    print("\nLoading CLIP model...")
    clip_model = SentenceTransformer(sentence_transformers_clip_model_name)

    test_query = "burger with cheese and lettuce"
    print(f"\nTest query: '{test_query}'")

    # Embed the text query using the CLIP model and normalize the embedding vector
    query_embedding = clip_model.encode(test_query, normalize_embeddings=True)
    query_vector = np.array(query_embedding, dtype=np.float32)

    top_k = 30

    # Query the index for top results
    distances, indices = faiss_index.search(query_vector, top_k=top_k)

    print(f"\nTop {top_k} most similar items to '{test_query}':")
    print("=" * 100)
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
        item_id = item_ids[idx]
        name, description = item_metadata_map.get(item_id, ("Unknown", "N/A"))

        # Format and display the result with name and description
        print(f"{rank}. [CLIP Score: {distance:.4f}]")
        print(f"   Item ID: {item_id}")
        print(f"   Name: {name}")
        print(f"   Description: {description}")
        print("-" * 100)
    print("=" * 100)

if __name__ == "__main__":

    # test_bm25_combined_description_index()

    # test_faiss_tags_hooks_index()

    # test_faiss_combined_description_index()
    
    test_faiss_clip_image_index()

