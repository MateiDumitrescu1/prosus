from paths_ import data_dir
csv_file_with_food_item_only_ids = f"{data_dir}/food_item_ids.csv"

def read_good_ids() -> list[str]:
    """
    Read good food item IDs from a CSV file.

    Returns:
        List of good food item IDs as strings.
    """
    good_ids = []
    with open(csv_file_with_food_item_only_ids, 'r', encoding='utf-8') as f:
        for line in f:
            item_id = line.strip()
            if item_id:
                good_ids.append(item_id)
    return good_ids

#! ---------------- TESTING ----------------
def test_read_good_ids():
    ids = read_good_ids()
    print(f"Read {len(ids)} good IDs")
    print("Sample IDs:", ids[:5])
    
    
if __name__ == "__main__":
    test_read_good_ids()
    print("Done testing read_good_ids.py")