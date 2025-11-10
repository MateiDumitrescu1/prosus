# Translate item names and descriptions from 5k_items_curated.csv
# Save results to JSONL format with item_id, translated_name, translated_description

import pandas as pd
import json
import asyncio
import os
from datetime import datetime
from prosus.api_wrappers.llms.openai_ import translate_text
from paths_ import fivek_items_csv_path, translations_dir


def read_items_csv(file_path: str) -> pd.DataFrame:
    """
    Read the 5k items CSV file and parse the itemMetadata JSON column.

    Returns:
        DataFrame with columns: item_id, name, description
    """
    df = pd.read_csv(file_path)

    # Parse the itemMetadata JSON column
    df['metadata'] = df['itemMetadata'].apply(json.loads)

    # Extract item_id, name, and description
    items_data = []
    for _, row in df.iterrows():
        item_id = row['itemId']
        name = row['metadata'].get('name', '')
        description = row['metadata'].get('description', '')

        items_data.append({
            'item_id': item_id,
            'name': name,
            'description': description
        })

    return pd.DataFrame(items_data)


async def translate_items_batch(items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate item names and descriptions to English using parallel async API calls.
    Processes items in batches of 10 for better performance.

    Args:
        items_df: DataFrame with columns item_id, name, description

    Returns:
        DataFrame with added columns: translated_name, translated_description
    """
    batch_size = 10
    total_items = len(items_df)

    translated_names = []
    translated_descriptions = []

    print(f"Translating {total_items} items...")

    # Process items in batches
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch = items_df.iloc[i:batch_end]

        # Create translation tasks for names and descriptions in parallel
        name_tasks = [
            translate_text(name, target_language="English")
            for name in batch['name']
        ]
        description_tasks = [
            translate_text(desc, target_language="English")
            for desc in batch['description']
        ]

        # Execute all tasks in parallel
        all_tasks = name_tasks + description_tasks
        results = await asyncio.gather(*all_tasks)

        # Split results back into names and descriptions
        batch_size_actual = len(batch)
        batch_translated_names = results[:batch_size_actual]
        batch_translated_descriptions = results[batch_size_actual:]

        # Add to results lists
        translated_names.extend(batch_translated_names)
        translated_descriptions.extend(batch_translated_descriptions)

        # Progress update
        print(f"Processed {batch_end}/{total_items} items...")

    # Add translated columns to dataframe
    items_df['translated_name'] = translated_names
    items_df['translated_description'] = translated_descriptions

    return items_df


def save_to_jsonl(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to JSONL format with only item_id, translated_name, translated_description.

    Args:
        df: DataFrame containing the data to save
        output_path: Path where the JSONL file will be saved
    """
    # Select only the required columns
    output_df = df[['item_id', 'translated_name', 'translated_description']]

    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in output_df.iterrows():
            json_line = {
                'item_id': row['item_id'],
                'translated_name': row['translated_name'],
                'translated_description': row['translated_description']
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    print(f"Translations saved to: {output_path}")


#!  ------- RUN  -------
def translate_items(cap: int = None) -> pd.DataFrame:
    """
    Main function to translate items and save to JSONL.

    Args:
        cap: Limit the number of items to translate (for testing)

    Returns:
        DataFrame with translated items
    """
    # Read items from CSV
    print("Reading items from CSV...")
    items_df = read_items_csv(fivek_items_csv_path)

    # Apply cap if specified
    if cap is not None:
        items_df = items_df.head(cap)
        print(f"Processing only {cap} items (cap applied)")

    # Translate items
    translated_df = asyncio.run(translate_items_batch(items_df))

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f'translated_items_{timestamp}.jsonl'
    output_path = os.path.join(translations_dir, output_filename)

    # Save to JSONL
    save_to_jsonl(translated_df, output_path)

    return translated_df


#!  ------- TESTING  -------
def test_read_items():
    """Test reading and parsing items from CSV."""
    items_df = read_items_csv(fivek_items_csv_path)
    print(f"Total items: {len(items_df)}")
    print("\nFirst 3 items:")
    print(items_df.head(3).to_string())


def test_translate_items():
    """Test translation with a small sample."""
    print("Testing translation with 5 items...")
    translated_df = translate_items(cap=5)

    print("\nTranslated Items (first 5):")
    for idx, row in translated_df.iterrows():
        print(f"\n{idx + 1}. Item ID: {row['item_id']}")
        print(f"   Original Name: {row['name']}")
        print(f"   Translated Name: {row['translated_name']}")
        print(f"   Original Description: {row['description']}")
        print(f"   Translated Description: {row['translated_description']}")


if __name__ == "__main__":
    # test_read_items()
    test_translate_items()
