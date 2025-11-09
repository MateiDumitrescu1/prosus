# create and save embeddings for the food items and queries
# save in a folder with the name em_model_used_datetime
import json
import numpy as np
import asyncio
from pathlib import Path
from datetime import datetime
from prosus.api_wrappers.voyage_.voyage_api_wrapper import VoyageAIModelAPI, VoyageAIModels
from paths_ import embeddings_output_dir
#!
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


def embed_tags():
    # dedup tags
    # embed the unique tags
    # save a CSV with the list of unique tags and another file containing the embedings. put both of these in the same folder
    pass

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
    
if __name__ == "__main__":
    test_embed_food_item_data()
    print("Test completed.")