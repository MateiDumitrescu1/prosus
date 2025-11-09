# create new CSVs with the translation for the queries
# and the translation of the food items

# 1. read the CSV file
# 2. translate the queries
# 3. save the translateion fo the quries in a new CSV file

import pandas as pd
import asyncio
import os
from datetime import datetime
from prosus.api_wrappers.llms.openai import translate_text
from paths_ import queries_csv_path, translations_dir

def read_queries_from_CSV(file_path) -> list[str]:
    """
    Read queries from CSV and return as a list of strings.
    """
    df = pd.read_csv(file_path)
    return df['search_term_pt'].tolist()


async def translate_queries_to_english(queries: list[str]) -> list[str]:
    """
    Translate queries to English using parallel async API calls.
    Processes 10 queries in parallel at a time for better performance.
    """
    translated_queries = []
    batch_size = 10

    # Process queries in batches of 10
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]

        # Create tasks for parallel execution
        tasks = [translate_text(query, target_language="English") for query in batch]

        # Execute all tasks in the batch in parallel
        batch_results = await asyncio.gather(*tasks)

        # Add results to the translated queries list
        translated_queries.extend(batch_results)

    return translated_queries

#!  ------- RUN  -------
def translate_the_queries(cap: int = None, save_to_csv: bool = True) -> list[str]:
    """
    Translate queries and optionally save to CSV file.

    Args:
        cap: Limit the number of queries to translate (for testing)
        save_to_csv: Whether to save the translations to a CSV file

    Returns:
        List of translated queries
    """
    #* Read the full DataFrame to preserve structure
    df = pd.read_csv(queries_csv_path)
    print(f"Read {len(df)} queries from CSV.")

    if cap is not None:
        df = df.head(cap)

    # Extract queries for translation
    queries = df['search_term_pt'].tolist()

    # Translate queries
    translated_queries = asyncio.run(translate_queries_to_english(queries))

    # Add translations as a new column
    df['search_term_en'] = translated_queries

    # Save to CSV if requested
    if save_to_csv:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f'translated_queries_{timestamp}.csv'
        output_path = os.path.join(translations_dir, output_filename)

        # Save DataFrame to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Translations saved to: {output_path}")

    return translated_queries

#!  ------- TESTING  -------
def test_read_queries_from_CSV():
    queries = read_queries_from_CSV(file_path=queries_csv_path)
    print("Queries:", queries)

def test_translate_the_queries():
    """
    Test translation with a small sample and save to CSV.
    """
    cap = None
    #!
    print("Testing translation with 5 queries...")
    translated_queries = translate_the_queries(cap=cap, save_to_csv=True)
    print("\nTranslated Queries:")
    for i, query in enumerate(translated_queries, 1):
        print(f"{i}. {query}")

if __name__ == "__main__":
    # test_read_queries_from_CSV()
    test_translate_the_queries()