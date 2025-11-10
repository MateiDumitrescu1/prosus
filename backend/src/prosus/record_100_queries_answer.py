"""
Process all queries from the queries CSV file using the match_query pipeline.
Save the top 30 item IDs for each query to a timestamped JSONL file.
"""
import csv
import json
import asyncio
from datetime import datetime
from pathlib import Path
from prosus.match_query import match_query, PipelineParameters
from paths_ import queries_csv_path, q100_output_dir

async def process_all_queries(params=PipelineParameters()):
    """
    Process all queries from the queries CSV file using match_query with default parameters.

    For each query:
    - Run match_query with default PipelineParameters
    - Extract the top 30 item IDs from the results
    - Save the results to a JSONL file in q100_output_dir

    Output format (JSONL):
    {"query": "...", "item_ids": [...]}
    """
    # Read all queries from the CSV file
    queries = []
    with open(queries_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_text = row['search_term_pt'].strip()
            if query_text:  # Skip empty queries
                queries.append(query_text)

    print(f"Loaded {len(queries)} queries from {queries_csv_path}")

    # Create timestamped folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(q100_output_dir) / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)

    # Create output files inside the run folder
    output_file = run_folder / "results.jsonl"
    params_file = run_folder / "params.json"

    # Save the parameters used for this run
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write(params.to_json_string())
    print(f"Created run folder: {run_folder}")
    print(f"Saved pipeline parameters to: {params_file}")

    # Process each query and save results
    results_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, query in enumerate(queries, start=1):
            print(f"\n{'='*60}")
            print(f"Processing query {idx}/{len(queries)}: '{query}'")
            print(f"{'='*60}")

            try:
                # Run match_query with default parameters
                results = await match_query(query, params)

                # Extract top 30 item IDs (match_query already returns sorted by score)
                top_30_item_ids = [result['item_id'] for result in results[:30]]

                # Save to JSONL
                output_record = {
                    "query": query,
                    "item_ids": top_30_item_ids
                }
                f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                f.flush()  # Ensure data is written immediately

                results_count += 1
                print(f"✓ Saved {len(top_30_item_ids)} item IDs for query '{query}'")

            except Exception as e:
                print(f"✗ Error processing query '{query}': {e}")
                # Save error record
                error_record = {
                    "query": query,
                    "item_ids": [],
                    "error": str(e)
                }
                f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                f.flush()

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed {results_count}/{len(queries)} queries")
    print(f"Run folder: {run_folder}")
    print(f"  - Results: {output_file}")
    print(f"  - Parameters: {params_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    params_to_use = PipelineParameters(clip_image_index_score_multiplier=0.0)
    asyncio.run(process_all_queries(params=params_to_use))