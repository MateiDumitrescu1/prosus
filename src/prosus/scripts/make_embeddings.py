# create and save embeddings for the food items and queries
# save in a folder with the name em_model_used_datetime
import json
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels, default_voyage_ai_embedding_model
from paths_ import embeddings_output_dir, combined_descriptions_dir, tags_and_hooks_embeddings_dir

#! ---------------------- read relevant data ----------------------
def read_tags_from_jsonl(jsonl_path: str) -> list[str]:
    """
    Read all tags from a JSONL file containing food item data.

    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of all tags found in the file
    """
    all_tags = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            tags = item.get('tags', [])
            all_tags.extend(tags)
    return all_tags

def read_associated_keyword_hooks_from_jsonl(jsonl_path: str) -> list[str]:
    """
    Read all associated keyword hooks from a JSONL file containing food item data.

    Args:
        jsonl_path: Path to the JSONL file
        
    Returns:
        List of all associated keyword hooks found in the file
    """
    all_hooks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            hooks = item.get('associated_keyword_hooks', [])
            all_hooks.extend(hooks)
    return all_hooks

def read_combined_descriptions_from_folder() -> dict[str, str]:
    """
    Read all JSONL files in `combined_descriptions_dir` and return a mapping of item_id to combined description.
    Dedup by itemId.
    """
    combined_descriptions = {}
    seen_ids = set() #* for deduplication
    
    for jsonl_file in Path(combined_descriptions_dir).glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item_id = item.get('itemId')
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                description = item.get('combined_description', '')
                combined_descriptions[item_id] = description
                
    return combined_descriptions

#! ---------------------- embed ----------------------
def embed_combined_descriptions(
        save_dir: str,
        model_name: VoyageAIModels = default_voyage_ai_embedding_model,
        cap_items : int | None = None
    ):
    """
    Create embeddings for food items using combined descriptions from the folder and save them to a JSONL file.
    Each row in the JSONL file contains itemId and the embedding vector.

    Args:
        save_dir: Directory where embeddings will be saved
        model_name: VoyageAI model to use for embeddings
        cap_items: Optional limit on number of items to embed (for testing)

    Returns:
        Path to the folder where embeddings were saved
    """
    print(f"Reading combined descriptions from folder...")

    # Read combined descriptions from the folder
    combined_descriptions = read_combined_descriptions_from_folder()

    print(f"Loaded {len(combined_descriptions)} food items")

    # Prepare item IDs and texts to embed
    item_ids = []
    texts_to_embed = []

    for item_id, description in combined_descriptions.items():
        item_ids.append(item_id)
        texts_to_embed.append(description)

    # Cap items if specified
    if cap_items is not None:
        item_ids = item_ids[:cap_items]
        texts_to_embed = texts_to_embed[:cap_items]
        print(f"Capped to {cap_items} items for embedding")

    # Call the embedding API using the wrapper in document mode
    print(f"Creating embeddings using {model_name}...")
    api_wrapper = VoyageAIModelAPI(model_name=model_name, show_progress_bar=True)

    async def get_embeddings():
        """Async function to get embeddings from the API"""
        embeddings = await api_wrapper.aembed_documents(texts_to_embed)
        await api_wrapper.aclose_client()
        return embeddings

    # Run the async embedding function
    embeddings_list = asyncio.run(get_embeddings())

    # Save the embeddings in a timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder_name = f"em_{cap_items if cap_items else 'full'}_{model_name}_{timestamp}"
    save_folder_path = Path(save_dir) / save_folder_name
    save_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings to {save_folder_path}...")

    # Save embeddings as JSONL file where each row has itemId and embedding vector
    jsonl_output_path = save_folder_path / "embeddings.jsonl"
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for item_id, embedding in zip(item_ids, embeddings_list):
            row = {
                "itemId": item_id,
                "embedding": embedding
            }
            f.write(json.dumps(row) + '\n')

    # Save metadata about the embedding process
    metadata = {
        "model_name": model_name,
        "num_items": len(item_ids),
        "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
        "timestamp": timestamp,
        "source": "combined_descriptions_folder"
    }
    with open(save_folder_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Successfully created and saved embeddings")
    print(f"  - Processed {metadata['num_items']} items")
    print(f"  - Embedding dimension: {metadata['embedding_dimension']}")
    print(f"  - Saved to: {save_folder_path}")
    print(f"  - JSONL file: {jsonl_output_path}")

    return save_folder_path

#TODO the tags need to be cleaned a bit: remove underscores
def embed_tags_and_hooks(
        save_dir: str,
        input_jsonl_path: str, 
        model_name: VoyageAIModels = default_voyage_ai_embedding_model
    ):
    """
    Create embeddings for all unique tags from a JSONL file and save them to disk.

    Args:
        input_jsonl_path: Path to the JSONL file containing food item data
        model_name: VoyageAI model to use for embeddings

    Returns:
        Path to the folder where tag embeddings were saved
    """
    print(f"Reading tags and associated keyword hooks from {input_jsonl_path}...")

    #* Read tags and associated keyword hooks using existing methods
    all_tags = read_tags_from_jsonl(input_jsonl_path)
    all_hooks = read_associated_keyword_hooks_from_jsonl(input_jsonl_path)

    #* Combine tags and hooks into a single list
    all_tags_and_hooks = all_tags + all_hooks

    print(f"Collected {len(all_tags)} tags and {len(all_hooks)} keyword hooks ({len(all_tags_and_hooks)} total)")

    #* Deduplicate tags and hooks while preserving order
    unique_tags = list(set(all_tags_and_hooks))
    unique_tags.sort()  # Sort for consistent ordering

    print(f"Found {len(unique_tags)} unique tags and keyword hooks")

    #TODO later on, filter the tags again and remove pairs that are too similar, eg. low edit distance (example: "gluten-free" vs "gluten free", "mcdonalds" vs "mcdonald's")
    
    #* Embed the unique tags using the VoyageAI wrapper
    print(f"Creating embeddings using {model_name}...")
    api_wrapper = VoyageAIModelAPI(model_name=model_name, show_progress_bar=True)

    async def get_embeddings():
        """Async function to get embeddings from the API"""
        embeddings = await api_wrapper.aembed_documents(unique_tags)
        await api_wrapper.aclose_client()
        return embeddings

    # Run the async embedding function
    embeddings_list = asyncio.run(get_embeddings())

    #* Save to save_dir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder_name = f"tag_embeddings_{model_name}_{timestamp}"
    save_folder_path = Path(save_dir) / save_folder_name
    save_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving tag embeddings to {save_folder_path}...")

    #* Save tag embeddings as JSONL file where each row has the tag/hook word and embedding vector
    jsonl_output_path = save_folder_path / "embeddings.jsonl"
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for tag, embedding in zip(unique_tags, embeddings_list):
            row = {
                "word": tag,
                "embedding": embedding
            }
            f.write(json.dumps(row) + '\n')

    #* Save metadata about the embedding process
    metadata = {
        "model_name": model_name,
        "num_unique_tags": len(unique_tags),
        "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
        "timestamp": timestamp,
        "source_file": input_jsonl_path
    }
    with open(save_folder_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    #* Save tags list separately for easy reference
    with open(save_folder_path / "tags_list.json", 'w', encoding='utf-8') as f:
        json.dump(unique_tags, f, indent=2)

    print(f"✓ Successfully created and saved tag embeddings")
    print(f"  - Processed {metadata['num_unique_tags']} unique tags")
    print(f"  - Embedding dimension: {metadata['embedding_dimension']}")
    print(f"  - Saved to: {save_folder_path}")
    print(f"  - JSONL file: {jsonl_output_path}")

    return save_folder_path

#! --- RUN ---

def run_embed_tags_and_hooks():
    """Run the tag and hook embedding function with default parameters"""
    jsonl_path = "../../../data/5k_items_new_format@v1.jsonl"
    embed_tags_and_hooks(
        save_dir=tags_and_hooks_embeddings_dir,
        input_jsonl_path=jsonl_path,
    )

def run_embed_combined_descriptions():
    """Run the combined descriptions embedding function with default parameters"""
    embed_combined_descriptions(
        save_dir=Path(embeddings_output_dir) / "combined_description_embeddings",
        cap_items=None  # Set to an integer for testing with fewer items
    )

#! --- TESTING ---

# def test_embed_food_item_data():
#     """Test the embedding creation function with combined descriptions from folder"""
#     save_directory = embeddings_output_dir + "/food_items" # Directory to save embeddings

#     # Call the embedding function
#     embed_food_item_data(
#         save_dir=save_directory,
#         cap_items=100
#     )

def test_read_combined_descriptions_from_folder():
    """Test reading combined descriptions from the folder"""
    combined_descriptions = read_combined_descriptions_from_folder()
    print(f"Read {len(combined_descriptions)} combined descriptions.")
    # Print first 5 entries
    for i, (item_id, description) in enumerate(combined_descriptions.items()):
        if i >= 5:
            break
        print(f"Item ID: {item_id}, Description: {description[:100]}...")

if __name__ == "__main__":
    # test_embed_food_item_data()
    # test_read_combined_descriptions_from_folder()
    run_embed_tags_and_hooks()
    # run_embed_combined_descriptions()
    print("Test completed.")