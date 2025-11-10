from paths_ import embeddings_output_dir, clip_image_embeddings_dir
from pathlib import Path
import csv
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
from prosus.constants import sentence_transformers_clip_image__model_name

#! ---------------------- READ DATA ----------------------
def read_item_images_from_csv(csv_path: str, images_base_dir: str) -> list[dict]:
    """
    Read food item image data from a CSV file containing itemMetadata.
    Converts CSV image paths to actual file paths in the downloaded_images directory.

    Args:
        csv_path: Path to the CSV file (5k_items_curated.csv format)
        images_base_dir: Base directory where downloaded images are stored

    Returns:
        A list of dictionaries, each containing:
            - item_id: str (the itemId from CSV)
            - image_file_paths: list of Path objects (actual file paths to downloaded images)
    """
    data = []
    images_dir = Path(images_base_dir)

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Parse itemMetadata JSON
            metadata = json.loads(row['itemMetadata'])

            # Extract and convert image paths to actual file paths
            image_paths_in_csv = metadata.get('images', [])
            actual_image_files = []

            for img_path in image_paths_in_csv:
                # CSV format: "0218a620-0e9a-4641-b4f9-72c236dcb5bb/202501141621_4T03_i.jpg"
                # File format: "0218a620-0e9a-4641-b4f9-72c236dcb5bb_202501141621_4T03_i.jpg"
                # Simply replace the '/' with '_'

                actual_filename = img_path.replace('/', '_')
                actual_file_path = images_dir / actual_filename

                # Check if file exists
                if actual_file_path.exists():
                    actual_image_files.append(actual_file_path)

            item = {
                'item_id': row['itemId'],
                'image_file_paths': actual_image_files
            }
            data.append(item)

    return data

#! ---------------------- EMBED ----------------------

def embed_images_using_clip(
        original_csv_path: str,
        save_dir: str,
        model_name: str = sentence_transformers_clip_image__model_name,
        images_base_dir: str | None = None,
        cap_items: int | None = None
    ):
    """
    Create embeddings for images only using CLIP model.
    Each image gets its own embedding saved as a separate row with the same itemId.
    If an item has multiple images, each image will be a separate row in the JSONL output.

    Args:
        original_csv_path: Path to the CSV file containing food item data with images.
        save_dir: Directory where embeddings will be saved.
        model_name: CLIP model to use (default: "clip-ViT-B-32")
        images_base_dir: Base directory where downloaded images are stored. If None, defaults to data/downloaded_images
        cap_items: Optional limit on number of items to embed (for testing)

    Returns:
        Path to the folder where embeddings were saved
    """
    print(f"Reading CSV file from {original_csv_path}...")

    # Set default images directory if not provided
    if images_base_dir is None:
        csv_dir = Path(original_csv_path).parent
        images_base_dir = str(csv_dir / "downloaded_images")

    print(f"Using images from: {images_base_dir}")

    # Read data using the helper method
    data = read_item_images_from_csv(original_csv_path, images_base_dir)

    print(f"Loaded {len(data)} food items")

    # Cap items if specified
    if cap_items is not None:
        data = data[:cap_items]
        print(f"Capped to {cap_items} items for embedding")

    # Load CLIP model
    print(f"Loading CLIP model: {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Prepare data structures for results
    # Each row will have: itemId, embedding, image_index, image_path
    embeddings_data = []

    # Statistics tracking
    stats = {
        "items_with_images": 0,
        "items_without_images": 0,
        "total_images_embedded": 0,
        "total_images_failed": 0,
        "total_embedding_rows": 0
    }

    print("Processing items and creating image embeddings...")

    # Process each item
    for item in tqdm(data, desc="Embedding items"):
        item_id = item['item_id']
        image_file_paths = item['image_file_paths']

        # Encode images if available
        if image_file_paths and len(image_file_paths) > 0:
            stats["items_with_images"] += 1

            for img_index, img_path in enumerate(image_file_paths):
                try:
                    # Load and encode the image
                    img = Image.open(img_path)
                    # Normalize embeddings for proper cosine similarity
                    img_embedding = model.encode(img, convert_to_tensor=False, normalize_embeddings=True)

                    # Save each image embedding as a separate row
                    embeddings_data.append({
                        "itemId": item_id,
                        "embedding": img_embedding.tolist(),
                        "image_index": img_index,
                        "image_path": str(img_path)
                    })

                    stats["total_images_embedded"] += 1
                    stats["total_embedding_rows"] += 1
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    stats["total_images_failed"] += 1
        else:
            stats["items_without_images"] += 1

    # Save the embeddings in a timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder_name = f"em_{cap_items if cap_items else 'full'}_{model_name.replace('/', '_')}_{timestamp}"
    save_folder_path = Path(save_dir)
    save_folder_path.mkdir(parents=True, exist_ok=True)
    save_folder_path = save_folder_path / save_folder_name
    save_folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings to {save_folder_path}...")

    # Save embeddings as JSONL file where each row has itemId, embedding, image_index, and image_path
    jsonl_output_path = save_folder_path / "embeddings.jsonl"
    with open(jsonl_output_path, 'w', encoding='utf-8') as f:
        for row in embeddings_data:
            f.write(json.dumps(row) + '\n')

    # Save metadata about the embedding process
    metadata = {
        "model_name": model_name,
        "num_items_processed": len(data),
        "num_embedding_rows": len(embeddings_data),
        "embedding_dimension": len(embeddings_data[0]["embedding"]) if embeddings_data else 0,
        "timestamp": timestamp,
        "source_csv": original_csv_path,
        "embedding_type": "images_only",
        "statistics": stats
    }
    with open(save_folder_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Successfully created and saved CLIP image embeddings")
    print(f"  - Processed {metadata['num_items_processed']} items")
    print(f"  - Total embedding rows: {metadata['num_embedding_rows']}")
    print(f"  - Embedding dimension: {metadata['embedding_dimension']}")
    print(f"  - Items with images: {stats['items_with_images']}")
    print(f"  - Items without images: {stats['items_without_images']}")
    print(f"  - Images successfully embedded: {stats['total_images_embedded']}")
    print(f"  - Images failed: {stats['total_images_failed']}")
    print(f"  - Saved to: {save_folder_path}")
    print(f"  - JSONL file: {jsonl_output_path}")

    return save_folder_path

#! ---------------- RUN ----------------

def run_embed_description_and_images_using_clip():
    """Run the embedding function for descriptions and images using CLIP model"""
    original_csv_path = "../../../data/5k_items_curated.csv"
    save_dir = clip_image_embeddings_dir
    
    cap_items_to_use = None
    
    embed_images_using_clip(
        original_csv_path=original_csv_path,
        save_dir=save_dir,
        cap_items=cap_items_to_use  # Set to an integer for testing with fewer items
    )

if __name__ == "__main__":
    run_embed_description_and_images_using_clip()
    print("Embedding completed.")
