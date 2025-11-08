# create new CSVs with the translation for the queries
# and the translation of the food items

# 1. read the CSV file
# 2. translate the queries 
# 3. save the translateion fo the quries in a new CSV file

import pandas as pd
import asyncio
from prosus.llm_api_wrapper import translate_text

from paths_ import queries_csv_path
def read_queries_from_CSV(file_path) -> list[str]:
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
def translate_the_queries(cap: int = None) -> list[str]:
    queries = read_queries_from_CSV(file_path=queries_csv_path)
    if cap is not None:
        queries = queries[:cap]

    translated_queries = asyncio.run(translate_queries_to_english(queries))
    return translated_queries

#!  ------- TESTING  -------
def test_read_queries_from_CSV():
    queries = read_queries_from_CSV(file_path=queries_csv_path)
    print("Queries:", queries)

def test_translate_the_queries():
    translated_queries = translate_the_queries(cap=5)
    print("Translated Queries:", translated_queries)

if __name__ == "__main__":
    # test_read_queries_from_CSV()
    test_translate_the_queries()