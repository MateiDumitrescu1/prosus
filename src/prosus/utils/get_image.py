"""
Script to download images from iFood static URLs.

The image URLs follow the format:
https://static.ifood-static.com.br/image/upload/t_low/pratos/<image_str>

Where image_str comes from the CSV file's itemMetadata.images field.
Example: 820af392-002c-47b1-bfae-d7ef31743c7f/202402200931_gxgyfoywbcj.jpeg
"""

import os
import requests
from pathlib import Path
from typing import Optional

from paths_ import downloaded_images_dir

# URL prefix for iFood images
IMAGE_URL_PREFIX = "https://static.ifood-static.com.br/image/upload/t_low/pratos/"


def download_image(
    image_str: str,
    output_folder: str = downloaded_images_dir,
    timeout: int = 10
) -> Optional[str]:
    """
    Download an image from iFood's static server and save it locally.

    Args:
        image_str: The image path from the CSV (e.g., "820af392-002c-47b1-bfae-d7ef31743c7f/202402200931_gxgyfoywbcj.jpeg")
        output_folder: Directory where images will be saved (default: "downloaded_images")
        timeout: Request timeout in seconds (default: 10)

    Returns:
        Path to the saved image file, or None if download failed

    Example:
        >>> download_image("820af392-002c-47b1-bfae-d7ef31743c7f/202402200931_gxgyfoywbcj.jpeg")
        'downloaded_images/820af392-002c-47b1-bfae-d7ef31743c7f_202402200931_gxgyfoywbcj.jpeg'
    """
    # Construct full URL
    full_url = f"{IMAGE_URL_PREFIX}{image_str}"

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate safe filename by replacing forward slashes with underscores
    safe_filename = image_str.replace("/", "_")
    output_file = output_path / safe_filename

    try:
        print(f"Downloading image from: {full_url}")

        # Download the image
        response = requests.get(full_url, timeout=timeout, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Save the image to disk
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully saved image to: {output_file}")
        return str(output_file)

    except requests.exceptions.Timeout:
        print(f"Timeout error while downloading {full_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {full_url}: {e}")
        return None
    except IOError as e:
        print(f"Error saving image to {output_file}: {e}")
        return None


def main():
    """
    Example usage of the download_image function.
    """
    # Example image strings from the CSV
    example_images = [
        "820af392-002c-47b1-bfae-d7ef31743c7f/202210182253_3h93mu9eg9y.jpg",
        "820af392-002c-47b1-bfae-d7ef31743c7f/202402200931_gxgyfoywbcj.jpeg",
    ]

    for image_str in example_images:
        result = download_image(image_str)
        if result:
            print(f"✓ Downloaded: {result}")
        else:
            print(f"✗ Failed to download: {image_str}")


if __name__ == "__main__":
    main()
