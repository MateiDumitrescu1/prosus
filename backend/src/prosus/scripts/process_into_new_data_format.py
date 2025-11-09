"""
Process 5k_items CSV file and transform it into the new JSONL data format.

This script reads the 5k_items_curated.csv file and creates a new JSONL file
with the streamlined data format as described in the README.md.
"""

import csv
import json
from pathlib import Path


def extract_tags_from_taxonomy(taxonomy):
    """
    Extract tags from the taxonomy field.

    Args:
        taxonomy: Dictionary containing l0, l1, l2 taxonomy levels

    Returns:
        List of tag strings extracted from taxonomy levels
    """
    tags = []
    if taxonomy:
        # Extract taxonomy levels and add as tags
        for level in ['l0', 'l1', 'l2']:
            if level in taxonomy and taxonomy[level]:
                tags.append(taxonomy[level])
    return tags


def extract_metrics(item_profile):
    """
    Extract metrics data including total orders and reorder rate.

    Args:
        item_profile: Item profile dictionary containing metrics data

    Returns:
        Dictionary containing total_orders and reorder_rate
    """
    metrics_data = item_profile.get('metrics', {})
    return {
        "total_orders": metrics_data.get('total_orders', 0),
        "reorder_rate": metrics_data.get('reorderRate', 0.0)
    }


def extract_dietary_flags(item_metadata):
    """
    Extract dietary flags from item metadata.

    Args:
        item_metadata: Item metadata dictionary containing dietary information

    Returns:
        Dictionary containing vegan, lactose_free, and organic flags
    """
    return {
        "vegan": item_metadata.get('vegan', False),
        "lactose_free": item_metadata.get('lacFree', False),
        "organic": item_metadata.get('organic', False)
    }


def extract_associated_keyword_hooks(search_data):
    """
    Extract associated keyword hooks from the search field.

    Args:
        search_data: List of search objects containing term, method, and count

    Returns:
        List of keyword strings (just the terms)
    """
    keyword_hooks = []
    if search_data:
        for search_item in search_data:
            if 'term' in search_item and search_item['term']:
                keyword_hooks.append(search_item['term'])
    return keyword_hooks


def extract_co_purchased_items(co_purchase_items):
    """
    Extract co-purchased items in the new simplified format.

    Args:
        co_purchase_items: List of co-purchase item objects

    Returns:
        List of dictionaries containing only item_id
    """
    items = []
    if co_purchase_items:
        for item in co_purchase_items:
            if 'item_id' in item:
                items.append({"item_id": item['item_id']})
    return items


def transform_item(row):
    """
    Transform a CSV row into the new data format.

    Args:
        row: Dictionary representing a CSV row

    Returns:
        Dictionary in the new streamlined format
    """
    # Parse JSON fields
    item_metadata = json.loads(row['itemMetadata'])
    item_profile = json.loads(row['itemProfile'])

    # Extract basic fields
    name = item_metadata.get('name', '')
    description = item_metadata.get('description', '')

    # Extract metrics (total orders and reorder rate)
    metrics = extract_metrics(item_profile)

    # Extract tags from taxonomy
    taxonomy = item_metadata.get('taxonomy', {})
    tags = extract_tags_from_taxonomy(taxonomy)

    # Add category_name from CSV to tags if present
    if 'category_name' in row and row['category_name']:
        tags.append(row['category_name'])

    # Extract dietary flags
    dietary_flags = extract_dietary_flags(item_metadata)

    # Extract associated keyword hooks from search data
    search_data = item_profile.get('search', [])
    associated_keyword_hooks = extract_associated_keyword_hooks(search_data)

    # Extract co-purchased items
    metrics_data = item_profile.get('metrics', {})
    co_purchase_data = metrics_data.get('coPurchaseItems', [])
    co_purchased_items = extract_co_purchased_items(co_purchase_data)

    # Create the new data structure
    new_item = {
        "item_id": row['itemId'],
        "name": name,
        "description": description,
        "metrics": metrics,
        "tags": tags,
        "dietary_flags": dietary_flags,
        "associated_keyword_hooks": associated_keyword_hooks,
        "co_purchased_items": co_purchased_items
    }

    return new_item


def process_csv_to_jsonl(input_csv_path, output_jsonl_path):
    """
    Process the CSV file and convert it to JSONL format.

    Args:
        input_csv_path: Path to the input CSV file
        output_jsonl_path: Path to the output JSONL file
    """
    items_processed = 0

    # Read CSV and write JSONL
    with open(input_csv_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                try:
                    # Transform the item
                    new_item = transform_item(row)

                    # Write as JSON line
                    jsonl_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                    items_processed += 1

                    # Progress indicator
                    if items_processed % 1000 == 0:
                        print(f"Processed {items_processed} items...")

                except Exception as e:
                    print(f"Error processing item {row.get('itemId', 'unknown')}: {e}")
                    continue

    print(f"\nProcessing complete! Total items processed: {items_processed}")
    print(f"Output file: {output_jsonl_path}")


def main():
    """Main entry point for the script."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent.parent
    input_csv = project_root / "data" / "5k_items_curated.csv"
    output_jsonl = project_root / "data" / "5k_items_new_format.jsonl"

    # Verify input file exists
    if not input_csv.exists():
        print(f"Error: Input file not found at {input_csv}")
        return

    print(f"Processing {input_csv}...")
    print(f"Output will be saved to {output_jsonl}")
    print()

    # Process the file
    process_csv_to_jsonl(input_csv, output_jsonl)

if __name__ == "__main__":
    main()
