import os, sys
#! this file defines important folder paths used across the project

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(current_dir, '..', ))

def check_dir_exists(dir_path: str):
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")

if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

#! data dir
data_dir = os.path.abspath(os.path.join(project_root_dir, 'data'))
check_dir_exists(data_dir)

queries_csv_path = os.path.abspath(os.path.join(data_dir, 'queries.csv'))
check_dir_exists(queries_csv_path)

fivek_items_csv_path = os.path.abspath(os.path.join(data_dir, '5k_items_curated.csv'))
check_dir_exists(fivek_items_csv_path)

new_data_format_dir = os.path.abspath(os.path.join(data_dir, 'new_data_format'))
check_dir_exists(new_data_format_dir)

#! main dirs in the `data` folder
downloaded_images_dir = os.path.abspath(os.path.join(data_dir, 'downloaded_images'))
check_dir_exists(downloaded_images_dir)

embeddings_output_dir = os.path.abspath(os.path.join(data_dir, 'embeddings_output'))
check_dir_exists(embeddings_output_dir)

combined_descriptions_dir = os.path.abspath(os.path.join(data_dir, 'combined_descriptions'))
check_dir_exists(combined_descriptions_dir)

translations_dir = os.path.abspath(os.path.join(data_dir, 'translations'))
check_dir_exists(translations_dir)

tags_and_hooks_embeddings_dir = os.path.abspath(os.path.join(embeddings_output_dir, 'tags_and_hooks_embeddings'))
check_dir_exists(tags_and_hooks_embeddings_dir)

clip_image_embeddings_dir = os.path.abspath(os.path.join(embeddings_output_dir, 'clip_image_embeddings'))
check_dir_exists(clip_image_embeddings_dir)

matching_output_dir = os.path.abspath(os.path.join(data_dir, 'matching_output')) # used to save the intermediate scores and final matching results
check_dir_exists(matching_output_dir)

#! ---------------- TESTING ----------------

def test_():
    print(f"current dir { current_dir } ")
    print(f"project root dir { project_root_dir } ")
    print(f"data dir { data_dir } ")
    print(f"downloaded images dir { downloaded_images_dir } ")

if __name__ == '__main__':
    test_()