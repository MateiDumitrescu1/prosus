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