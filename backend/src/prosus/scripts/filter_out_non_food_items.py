"""
Script to filter food-related items from a CSV file using LLM classification.

This script reads a CSV file containing item data, extracts the name and description
of each item, queries an LLM to determine if the item is food-related, and saves
the item IDs of food-related items to a new CSV file.
"""

import csv
import json
import asyncio
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path to import the LLM wrapper
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from prosus.api_wrappers.llms.openai_ import get_openai_llm_response, OpenAIModel


async def is_food_related(item_name: str, item_description: str) -> bool:
    """
    Query the LLM to determine if an item is food-related.

    Args:
        item_name: Name of the item
        item_description: Description of the item

    Returns:
        True if the item is food-related, False otherwise
    """
    prompt = f"""
        You are classifying grocery/retail items to determine if they are food-related or not.

        Item Name: {item_name}
        Item Description: {item_description}

        Is this item food-related? 
        Respond with ONLY "YES" if it's food-related or "NO" if it's not food-related. Do not provide any explanation or additional text.
    """

    try:
        response = await get_openai_llm_response(prompt, model=OpenAIModel.GPT_4O_MINI)
        response = response.strip().upper()

        # Handle various possible responses
        if "YES" in response:
            return True
        elif "NO" in response:
            return False
        else:
            # If response is unclear, default to False (not food-related)
            print(f"Unclear response for '{item_name}': {response}. Defaulting to NO.")
            return False
    except Exception as e:
        print(f"Error processing '{item_name}': {e}. Defaulting to NO.")
        return False


async def process_item(row: dict) -> tuple[str, bool, str]:
    """
    Process a single item to determine if it's food-related.

    Args:
        row: Dictionary containing item data from CSV

    Returns:
        Tuple of (item_id, is_food, item_name)
    """
    try:
        item_id = row['itemId']
        item_metadata_str = row['itemMetadata']

        # Parse the JSON metadata
        item_metadata = json.loads(item_metadata_str)
        item_name = item_metadata.get('name', '')
        item_description = item_metadata.get('description', '')

        # Query the LLM
        is_food = await is_food_related(item_name, item_description)

        return (item_id, is_food, item_name)

    except Exception as e:
        print(f"Error processing row: {e}")
        return (None, False, "Error")


async def process_csv(
    input_csv_path: str,
    output_csv_path: str = None,
    limit: int = None,
    batch_size: int = 20
) -> None:
    """
    Process a CSV file to filter food-related items using concurrent batch processing.

    Args:
        input_csv_path: Path to the input CSV file
        output_csv_path: Path to the output CSV file (optional, auto-generated if not provided)
        limit: Maximum number of items to process (optional, processes all if not provided)
        batch_size: Number of items to process concurrently (default: 20)
    """
    # Generate output path if not provided
    if output_csv_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_csv_path = f"data/food_items_{timestamp}.csv"

    # Create output directory if it doesn't exist
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    food_item_ids = []
    processed_count = 0
    food_count = 0
    non_food_count = 0

    print(f"Reading CSV from: {input_csv_path}")
    print(f"Output will be saved to: {output_csv_path}")
    print(f"Processing with batch size: {batch_size}")

    # Read the CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        # Apply limit if specified
        if limit:
            rows = rows[:limit]
            print(f"Processing {limit} items...")
        else:
            print(f"Processing {len(rows)} items...")

        # Process items in batches
        with tqdm(total=len(rows), desc="Classifying items") as pbar:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]

                # Process batch concurrently
                results = await asyncio.gather(*[process_item(row) for row in batch])

                # Collect results
                for item_id, is_food, item_name in results:
                    if item_id is None:  # Error occurred
                        continue

                    if is_food:
                        food_item_ids.append(item_id)
                        food_count += 1
                        print(f"FOOD: {item_name}")
                    else:
                        non_food_count += 1
                        print(f"NOT FOOD: {item_name}")

                    processed_count += 1

                # Update progress bar
                pbar.update(len(batch))

    # Write the food item IDs to output CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['itemId'])  # Header
        for item_id in food_item_ids:
            writer.writerow([item_id])

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total items processed: {processed_count}")
    print(f"Food-related items: {food_count} ({food_count/processed_count*100:.1f}%)")
    print(f"Non-food items: {non_food_count} ({non_food_count/processed_count*100:.1f}%)")
    print(f"\nFood item IDs saved to: {output_csv_path}")
    print("=" * 60)


async def main():
    """Main function to run the script."""
    # Configuration
    input_csv_path = "../../../data/5k_items_curated.csv"
    output_csv_path = None  # Will be auto-generated with timestamp
    limit = None  # Set to a number to test on a subset, None to process all

    # Uncomment the line below to test on a small subset first
    # limit = 10

    await process_csv(input_csv_path, output_csv_path, limit)


if __name__ == "__main__":
    asyncio.run(main())
