import base64
import mimetypes, sys
from pathlib import Path
from typing import Optional


def image_to_base64(image_path: str, include_data_uri: bool = False) -> str:
    """
    Convert an image file to base64 encoded string.

    Args:
        image_path: Path to the image file
        include_data_uri: If True, includes the data URI prefix (e.g., 'data:image/png;base64,')
                         for direct use in HTML/CSS. Default is False.

    Returns:
        Base64 encoded string representation of the image

    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the file is not a valid image format
    """
    # Validate that the file exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read the image file in binary mode
    with open(image_path, 'rb') as image_file_handle:
        image_binary_data = image_file_handle.read()

    # Encode the binary data to base64
    base64_encoded = base64.b64encode(image_binary_data).decode('utf-8')

    # Optionally add data URI prefix for web usage
    if include_data_uri:
        # Detect the MIME type of the image
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None or not mime_type.startswith('image/'):
            raise ValueError(f"File does not appear to be a valid image: {image_path}")

        # Create the data URI format
        base64_encoded = f"data:{mime_type};base64,{base64_encoded}"

    return base64_encoded


def base64_to_image(base64_string: str, output_path: str) -> None:
    """
    Convert a base64 encoded string back to an image file.

    Args:
        base64_string: Base64 encoded string (with or without data URI prefix)
        output_path: Path where the image file should be saved

    Raises:
        ValueError: If the base64 string is invalid
    """
    # Remove data URI prefix if present
    if base64_string.startswith('data:'):
        # Extract just the base64 portion after 'base64,'
        base64_string = base64_string.split('base64,', 1)[1]

    # Decode the base64 string to binary data
    try:
        image_binary_data = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Invalid base64 string: {e}")

    # Write the binary data to file
    with open(output_path, 'wb') as output_file:
        output_file.write(image_binary_data)
