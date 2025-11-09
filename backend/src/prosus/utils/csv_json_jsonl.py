import json, os, csv

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

def combine_jsonl_files(input_files: list[str], output_file: str) -> None:
    """
    Combine multiple JSONL files into one.

    Args:
        input_files: List of paths to input JSONL files
        output_file: Path to the output combined JSONL file
    """
    combined_data = []

    for input_file in input_files:
        data = read_jsonl_to_list(input_file)
        combined_data.extend(data)

    with open(output_file, 'w', encoding='utf-8') as file:
        for item in combined_data:
            file.write(json.dumps(item) + '\n')

def list_jsonl_files_in_folder(folder_path: str) -> list[str]:
    """
    List all JSONL files in a given folder.

    Args:
        folder_path: Path to the folder

    Returns:
        List of JSONL file paths
    """
    jsonl_files = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, filename))
    return jsonl_files


#! ----------- RUN -----------

def run_combine_jsonl_files():
    all_current_files = list_jsonl_files_in_folder("../../../data/combined_descriptions")
    output_file = 'combined_descriptions@v0.jsonl'
    combine_jsonl_files(input_files=all_current_files, output_file=output_file)

if __name__ == "__main__":
    run_combine_jsonl_files()