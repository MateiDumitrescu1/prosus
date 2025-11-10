import json
from pathlib import Path
from paths_ import combined_descriptions_dir

def get_combined_descriptions_from_folder(version: str = None) -> dict[str, str]:
    """
    Get all JSONL files in `combined_descriptions_dir` and return a mapping of item_id to combined description.
    Dedup by itemId.

    Args:
        version: Optional version string (e.g., "v0", "v1"). If provided, only files ending with that version will be read.
        
    Returns:
        A dictionary mapping item IDs to their combined descriptions. Eg. {'item123': 'This is the combined description.', ...}
    """
    
    combined_descriptions = {}
    seen_ids = set() #* for deduplication

    # Determine glob pattern based on whether version is specified
    glob_pattern = f"*{version}.jsonl" if version is not None else "*.jsonl"

    for jsonl_file in Path(combined_descriptions_dir).glob(glob_pattern):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item_id = item.get('itemId')
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                description = item.get('combined_description', '')
                combined_descriptions[item_id] = description

    if combined_descriptions is None or len(combined_descriptions) == 0:
        raise ValueError(f"No combined descriptions found in folder '{combined_descriptions_dir}' with version '{version}'.")
    
    return combined_descriptions

def add_tags_to_combined_description(
        new_format_jsonl_file_path: str,
        combined_description_file_path: str,
    ):
    """
    To the "combined_description" field in the combined description JSONL file, append this string:
    "\n\nEtiquetas: [tag1, tag2, ...]"
    where [tag1, tag2, ...] are the tags from JSONL file.
    new format = check the README file: we reformated the way food items are stored.

    The content of the combined description file will be updated in-place.
    """

    # Step 1: Read the new format file and create a mapping of item_id to tags
    item_tags_map = {}
    with open(new_format_jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            item_id = item.get('item_id')
            tags = item.get('tags', [])
            if item_id:
                item_tags_map[item_id] = tags

    # Step 2: Read the combined description file and update each entry
    updated_entries = []
    with open(combined_description_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            item_id = entry.get('itemId')

            # Get tags for this item from the new format file
            tags = item_tags_map.get(item_id, [])

            # Format tags as a string: [tag1, tag2, ...]
            tags_string = ', '.join(tags)

            # Append tags to the combined_description
            current_description = entry.get('combined_description', '')
            entry['combined_description'] = f"{current_description}\n\nEtiquetas: [{tags_string}]"

            updated_entries.append(entry)

    # Step 3: Write the updated entries back to the file in-place
    with open(combined_description_file_path, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
#! ---- TESTING ----
def test_add_tags_to_combined_description():
    new_format_jsonl_file_path = "../../../../data/5k_items_new_format@v1.jsonl"
    combined_description_file_path = "../../../../data/combined_descriptions/combined_descriptions@v0.jsonl"
    add_tags_to_combined_description(
        new_format_jsonl_file_path=new_format_jsonl_file_path,
        combined_description_file_path=combined_description_file_path,
    )
    
if __name__ == "__main__":
    test_add_tags_to_combined_description()
    print("Tests completed!")