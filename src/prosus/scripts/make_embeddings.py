# create and save embeddings for the food items and queries
# save in a folder with the name em_model_used_datetime
import json
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels
from paths_ import embeddings_output_dir, combined_descriptions_dir

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
    """Read all JSONL files in `combined_descriptions_dir` and return a mapping of item_id to combined description."""
    combined_descriptions = {}
    for jsonl_file in Path(combined_descriptions_dir).glob("*.jsonl"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item_id = item.get('itemId')
                description = item.get('combined_description', '')
                combined_descriptions[item_id] = description
    return combined_descriptions

#! ---------------------- embed ----------------------
def embed_food_item_data(
        jsonl_path: str, 
        save_dir: str, 
        model_name: VoyageAIModels = VoyageAIModels.VOYAGE_3_5_LITE,
        cap_items : int | None = None
    ):
    """
    Create embeddings for food items from a JSONL file and save them to disk.

    Args:
        jsonl_path: Path to the JSONL file containing food item data
        save_dir: Directory where embeddings will be saved
        model_name: VoyageAI model to use for embeddings

    Returns:
        Path to the folder where embeddings were saved
    """
    print(f"Reading food items from {jsonl_path}...")

    # Read the JSONL file line by line
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))

    print(f"Loaded {len(items)} food items")
    
    if cap_items is not None:
        items = items[:cap_items]
        print(f"Capped to {cap_items} items for embedding")

    # Unify name + description into a single text string (simply append)
    item_ids = []
    texts_to_embed = []

    for item in items:
        item_ids.append(item['item_id'])

        # Extract name and description from the item
        name = item.get('name', '')
        description = item.get('description', '')

        # Combine name and description by simple concatenation
        combined_text = f"{name} {description}".strip()
        texts_to_embed.append(combined_text)

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

    # Convert embeddings to numpy array and save
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    np.save(save_folder_path / "embeddings.npy", embeddings_array)

    # Save item IDs in the same order as embeddings for mapping
    with open(save_folder_path / "item_ids.json", 'w', encoding='utf-8') as f:
        json.dump(item_ids, f, indent=2)

    # Save metadata about the embedding process
    metadata = {
        "model_name": model_name,
        "num_items": len(items),
        "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
        "timestamp": timestamp,
        "source_file": jsonl_path
    }
    with open(save_folder_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Successfully created and saved embeddings")
    print(f"  - Processed {metadata['num_items']} items")
    print(f"  - Embedding dimension: {metadata['embedding_dimension']}")
    print(f"  - Saved to: {save_folder_path}")

    return save_folder_path

def embed_tags(input_jsonl_path: str, model_name: VoyageAIModels = VoyageAIModels.VOYAGE_3_5_LITE):
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

    # Create a mapping from tag to embedding
    tag_embeddings = {}
    for tag, embedding in zip(unique_tags, embeddings_list):
        tag_embeddings[tag] = embedding

    #* Save to embeddings_output_dir with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder_name = f"tag_embeddings_{model_name}_{timestamp}"
    save_folder_path = Path(embeddings_output_dir) / "tags" / save_folder_name
    save_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving tag embeddings to {save_folder_path}...")

    #* Save tag embeddings as JSON (embeddings are already lists for JSON serialization)
    with open(save_folder_path / "tag_embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(tag_embeddings, f, indent=2)

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

    return save_folder_path

#! --- RUN ---

def run_embed_tags():
    """Run the tag embedding function with default parameters"""
    jsonl_path = "../../../data/5k_items_new_format@v1.jsonl"
    embed_tags(input_jsonl_path=jsonl_path, model_name=VoyageAIModels.VOYAGE_3_5_LITE)

#! --- TESTING ---

def test_embed_food_item_data():
    """Test the embedding creation function with sample data"""
    sample_jsonl_path = "../../../data/5k_items_new_format@v1.jsonl"  # Path to a sample JSONL file
    save_directory = embeddings_output_dir + "/food_items" # Directory to save embeddings

    # Call the embedding function
    embed_food_item_data(
        jsonl_path=sample_jsonl_path,
        save_dir=save_directory,
        model_name=VoyageAIModels.VOYAGE_3_5_LITE,
        cap_items=100
    )

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
    test_read_combined_descriptions_from_folder()
    print("Test completed.")