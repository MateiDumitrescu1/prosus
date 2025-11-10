"""
LLM Judge for evaluating search result relevance.

This script uses GPT-5-MINI to judge the relevance of search results for 100 queries.
For each query, it evaluates the returned items and counts how many are relevant vs irrelevant.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple
# from prosus.api_wrappers.llms.openai_ import get_openai_llm_response, OpenAIModel
from prosus.api_wrappers.llms.openrouter import OpenRouterModel, get_openrouter_llm_response
from prosus.scripts.script_utils.combined_description_utils import get_combined_descriptions_from_folder
from paths_ import data_dir
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configuration paths
combined_descriptions = get_combined_descriptions_from_folder(version="v1")

ITEMS_FILE = Path(data_dir) / "5k_items_new_format@v1.jsonl"
RESULTS_FILE = Path(data_dir) / "q100_output" / "run_20251110_052004" / "results.jsonl"
OUTPUT_FILE = Path(data_dir) / "q100_output" / "run_20251110_052004" / f"llm_judge_results_{current_time}.txt"

# Number of top items to judge per query
JUDGE_TOP_K = 10

# Number of queries to process concurrently
BATCH_SIZE = 10


def load_items() -> Dict[str, Dict[str, str]]:
    """
    Load all items from the JSONL file into a dictionary for quick lookup.

    Returns:
        Dictionary mapping item_id to item data (name, description)
    """
    items = {}
    with open(ITEMS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            items[item['item_id']] = {
                'name': item['name'],
                'description': item['description']
            }
    print(f"Loaded {len(items)} items from database")
    return items


def load_query_results() -> List[Dict]:
    """
    Load query results from the JSONL file.

    Returns:
        List of dictionaries containing query and item_ids
    """
    results = []
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    print(f"Loaded {len(results)} query results")
    return results


async def evaluate_query(query: str, item_ids: List[str], items_db: Dict[str, Dict[str, str]], combined_descriptions: Dict[str, str], top_k: int = JUDGE_TOP_K) -> Tuple[int, int, int, str]:
    """
    Evaluate top K items for a single query in one API call.

    Args:
        query: The search query
        item_ids: List of item IDs returned for this query (ordered by relevance)
        items_db: Dictionary of all items
        combined_descriptions: Dictionary mapping item_id to combined description
        top_k: Number of top items to evaluate

    Returns:
        Tuple of (relevant_count, somewhat_relevant_count, irrelevant_count, prompt)
    """
    # Take only top K items
    top_item_ids = item_ids[:top_k]

    # Build the items list for the prompt
    items_text = ""
    valid_count = 0
    for i, item_id in enumerate(top_item_ids, 1):
        if item_id in items_db:
            item = items_db[item_id]
            combined_desc = combined_descriptions.get(item_id, item['description'])
            if combined_desc is None or combined_desc.strip() == "":
                print("⚠️⚠️⚠️ Warning: Combined description is empty!")
                combined_desc = item['description']
                
            items_text += f"{i}. {item['name']} - {combined_desc}\n"
            valid_count += 1
        else:
            print(f"Warning: Item {item_id} not found in database")
            items_text += f"{i}. [Item not found]\n"

    # Construct the prompt - ask GPT to count relevant vs somewhat relevant vs irrelevant
    prompt = f"""Query: "{query}"

    Items returned by the search system:
    {items_text}
    Categorize each item into one of three categories:
    1. RELEVANT: The item is a good match for the query
    2. SOMEWHAT RELEVANT: The item's description or name mentions at least 1 detail or keyword from the query (e.g., "korean spicy chicken" and "Cup Noodles instant noodles in a cup, chicken flavor" share "chicken", so it's somewhat relevant)
    The keyword doesn't have to be an exact match. For example, "sandwich" and "burger" can be considered similar enough for this criteria.
    3. IRRELEVANT: The item doesn't match the query whatsoever and has nothing in common.

    Some queries are very specific, don't be too strict in your judgment.
    For example, if the query is "Carne assada lentamente com acompanhamentos", results of meat dishes with sides should be considered relevant.

    Answer ONLY with three numbers separated by spaces: relevant_count somewhat_relevant_count irrelevant_count
    Example: "5 2 3" means 5 relevant, 2 somewhat relevant, and 3 irrelevant.
    Do not include any explanation or other text.
    """

    # Get the response from GPT-5-MINI
    response = await get_openrouter_llm_response(prompt, model=OpenRouterModel.KIMI_K2)

    # Parse the three numbers from the response
    try:
        parts = response.strip().split()
        relevant_count = int(parts[0])
        somewhat_relevant_count = int(parts[1])
        irrelevant_count = int(parts[2])

        # Validate that the counts add up correctly
        total = relevant_count + somewhat_relevant_count + irrelevant_count
        if total != top_k:
            print(f"Warning: Counts ({relevant_count} + {somewhat_relevant_count} + {irrelevant_count} = {total}) don't add up to {top_k}. Response: {response}")
            # Adjust if needed
            if total > 0:
                # Scale proportionally
                relevant_count = round(relevant_count * top_k / total)
                somewhat_relevant_count = round(somewhat_relevant_count * top_k / total)
                irrelevant_count = top_k - relevant_count - somewhat_relevant_count
            else:
                # Default to all irrelevant if response is invalid
                relevant_count = 0
                somewhat_relevant_count = 0
                irrelevant_count = top_k

        return relevant_count, somewhat_relevant_count, irrelevant_count, prompt
    except (IndexError, ValueError) as e:
        print(f"Error parsing response: '{response}'. Error: {e}")
        # Default to all irrelevant if we can't parse
        return 0, 0, top_k, prompt


async def main():
    """
    Main function to run the LLM judge evaluation.
    """
    print("Starting LLM judge evaluation...")

    # Load data
    items_db = load_items()
    query_results = load_query_results()

    # Create folder for problematic prompts (queries with 5+ irrelevant matches)
    prompts_folder = Path(data_dir) / "q100_output" / "run_20251110_052004" / f"problematic_prompts_{current_time}"
    prompts_folder.mkdir(parents=True, exist_ok=True)

    # Store all results in order
    all_results = []

    # Process queries in batches
    total_queries = len(query_results)
    for batch_start in range(0, total_queries, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_queries)
        batch = query_results[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start//BATCH_SIZE + 1} (queries {batch_start + 1}-{batch_end})")
        print(f"{'='*60}")

        # Create tasks for all queries in this batch
        tasks = []
        for i, result in enumerate(batch):
            query = result['query']
            item_ids = result['item_ids']
            query_num = batch_start + i + 1

            print(f"[{query_num}/{total_queries}] Queuing: {query[:50]}...")

            # Create task for this query
            task = evaluate_query(query, item_ids, items_db, combined_descriptions)
            tasks.append((query_num, query, task))

        # Execute all tasks in this batch concurrently
        print(f"\nExecuting {len(tasks)} queries concurrently...")
        batch_results = await asyncio.gather(*[task for _, _, task in tasks])

        # Store results with their query info
        for (query_num, query, _), (relevant_count, somewhat_relevant_count, irrelevant_count, prompt) in zip(tasks, batch_results):
            all_results.append((query_num, query, relevant_count, somewhat_relevant_count, irrelevant_count, prompt))
            print(f"[{query_num}/{total_queries}] ✓ {relevant_count} relevant, {somewhat_relevant_count} somewhat relevant, {irrelevant_count} irrelevant - {query[:50]}...")

            # Save prompt if there are 5 or more irrelevant matches
            if irrelevant_count >= 5:
                prompt_file = prompts_folder / f"query_{query_num:03d}.txt"
                with open(prompt_file, 'w', encoding='utf-8') as pf:
                    pf.write(prompt)
                print(f"  ⚠️  Saved problematic prompt to: {prompt_file.name}")

    # Write all results to file in order
    print(f"\n{'='*60}")
    print("Writing results to file...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for query_num, query, relevant_count, somewhat_relevant_count, irrelevant_count, prompt in all_results:
            output_line = f"{relevant_count} {somewhat_relevant_count} {irrelevant_count}\n"
            out_f.write(output_line)

    print(f"✓ Evaluation complete! Results saved to: {OUTPUT_FILE}")

    # Count and report problematic prompts
    problematic_count = sum(1 for _, _, _, _, irrelevant_count, _ in all_results if irrelevant_count >= 5)
    if problematic_count > 0:
        print(f"⚠️  {problematic_count} queries had 5+ irrelevant matches. Prompts saved to: {prompts_folder}")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())