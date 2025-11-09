import csv
import json


def read_jsonl_to_list(file_path: str) -> list[dict]:
    """
    Read all lines from a JSONL (JSON Lines) file into a list of Python dictionaries.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one for each line in the JSONL file

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    data_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            try:
                # Parse JSON from the line and add to list
                data_dict = json.loads(line)
                data_list.append(data_dict)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_number}: {e.msg}",
                    e.doc,
                    e.pos
                )

    return data_list

