import os
import re
from pathlib import Path

from utils.input_utils import is_path_supported_format, supported_image_formats


def check_path_accessible(path: Path):
    """
    Check if the provide path is accessible, raise error for different checks
    Args:
        path (Path): Path to check
    Raises:
        TypeError: Path is not a Path object
        FileNotFoundError: Dir/file does not exist at location
        PermissionError: No read access for folder/file
    """
    if not isinstance(path, Path):
        raise TypeError(f"provided object {path} is not Path, but {type(path)}")
    if not path.exists():
        raise FileNotFoundError(f"Missing path: {path}")
    if not os.access(path=path, mode=os.R_OK):
        raise PermissionError(f"No access to {path} for read operations")

    return True


def image_path_to_xml_path(image_path: Path, check: bool = True) -> Path:
    """
    Return the corresponding xml path for a image

    Args:
        image_path (Path): Image path
        check (bool): Flag to turn off checking existence

    Returns:
        Path: XML path
    """
    xml_path = image_path.absolute().parent.joinpath("page", image_path.stem + ".xml")

    if check:
        check_path_accessible(xml_path)

    return xml_path


def xml_path_to_image_path(xml_path: Path, check: bool = True) -> Path:
    """
    Return the corresponding image path for an xml

    Args:
        xml_path (Path): XML path
        check (bool): Flag to turn off checking existence

    Raises:
        FileNotFoundError: No image for xml path

    Returns:
        Path: Image_path
    """

    # Check if image with name exist if so return, else raise Error
    image_path_dir = xml_path.absolute().parents[1]

    image_paths = image_path_dir.glob(f"{xml_path.stem}*")

    for image_path in image_paths:
        # TODO multiple images with the same name (extract from pageXML what to use)
        if is_path_supported_format(image_path, supported_image_formats):
            break
    else:
        raise FileNotFoundError(f"No image equivalent found for {xml_path}")

    if check:
        check_path_accessible(image_path)

    return image_path


def unique_path(path: str | Path, current_count: int = 1) -> Path:
    """
    Check if current path exists if it does recursively check if the next path with (n) added to the end already exists

    Args:
        path (str | Path): base path
        current_count (int, optional): addition if the path already exists. Defaults to 1.

    Returns:
        Path: a unique path
    """

    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return path

    if match := re.fullmatch(r"(.*)(\(\d+\))", path.stem):
        path_suggestion = Path(match.group(1) + f"({current_count})" + path.suffix)
    else:
        path_suggestion = Path(path.stem + f"({current_count})" + path.suffix)

    path_suggestion = path.parent.joinpath(path_suggestion)

    current_count = current_count + 1

    return unique_path(path_suggestion, current_count=current_count)
