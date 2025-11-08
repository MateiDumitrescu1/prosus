import csv

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