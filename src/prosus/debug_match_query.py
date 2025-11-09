"""
Debug utility to examine match query results by looking up item details from the CSV file.
"""
import json
import csv
from pathlib import Path
from typing import Optional
from paths_ import data_dir

def debug_match_results(results_folder_path: str, top_n: Optional[int] = None):
    """
    Analyzes the final_results.jsonl from a given results folder and prints
    the name and description of matched items from the original CSV file.

    Args:
        results_folder_path: Path to the matching_output results folder (e.g.,
                           "data/matching_output/matching_results_20251109_203933")
        top_n: Optional number of top results to display. If None, displays all results.
    """
    # Define paths
    csv_path = Path(data_dir) / "5k_items_curated.csv"
    results_folder = Path(results_folder_path)
    final_results_path = results_folder / "final_results.jsonl"

    # Validate paths
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    if not final_results_path.exists():
        print(f"Error: final_results.jsonl not found at {final_results_path}")
        return

    # Build lookup dictionary from CSV: itemId -> (name, description)
    print(f"Loading item data from {csv_path}...")
    item_lookup = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row['itemId']
            # Parse the itemMetadata JSON string
            try:
                metadata = json.loads(row['itemMetadata'])
                name = metadata.get('name', 'N/A')
                description = metadata.get('description', 'N/A')
                item_lookup[item_id] = {
                    'name': name,
                    'description': description
                }
            except json.JSONDecodeError:
                print(f"Warning: Could not parse metadata for item {item_id}")
                continue

    print(f"Loaded {len(item_lookup)} items from CSV")

    # Read final results
    print(f"\nReading results from {final_results_path}...")
    results = []
    with open(final_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            results.append(result)

    # Limit results if top_n is specified
    if top_n is not None:
        results = results[:top_n]

    # Print results with item details
    print(f"\n{'='*80}")
    print(f"MATCH QUERY RESULTS - Top {len(results)} Items")
    print(f"{'='*80}\n")

    for rank, result in enumerate(results, start=1):
        item_id = result['item_id']
        score = result['score']
        aggregated_score = result.get('aggregated_score', 'N/A')
        relevance_score = result.get('relevance_score', 'N/A')

        # Look up item details
        if item_id in item_lookup:
            item_data = item_lookup[item_id]
            name = item_data['name']
            description = item_data['description']
        else:
            name = "NOT FOUND IN CSV"
            description = "N/A"

        # Print formatted result
        print(f"Rank {rank}:")
        print(f"  Item ID: {item_id}")
        print(f"  Name: {name}")
        print(f"  Description: {description}")
        print(f"  Final Score: {score:.4f}")
        print(f"  Aggregated Score: {aggregated_score if isinstance(aggregated_score, str) else f'{aggregated_score:.4f}'}")
        print(f"  Relevance Score: {relevance_score if isinstance(relevance_score, str) else f'{relevance_score:.4f}'}")
        print(f"  {'-'*76}")

    print(f"\n{'='*80}")
    print(f"Total results displayed: {len(results)}")
    print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    # Example: Debug the most recent results
    results_folder = "../../data/matching_output/matching_results_20251109_203933"

    # Display top 10 results
    debug_match_results(results_folder, top_n=10)