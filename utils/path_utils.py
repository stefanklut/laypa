import os
from pathlib import Path
import re
from typing import Iterable, Optional
from natsort import os_sorted

def check_path_accessible(path: Path):
    """
    Check if the provide path is accessible, raise error for different checks
    Args:
        path (Path): path to check
    Raises:
        ValueError: path is not a Path object
        FileNotFoundError: folder/file does not exist at location
        PermissionError: no read access for folder/file
    """
    if not isinstance(path, Path):
        raise ValueError(f"provided object {path} is not Path, but {type(path)}")
    if not path.exists():
        raise FileNotFoundError(f"Missing path: {path}")
    if not os.access(path=path, mode=os.R_OK):
        raise PermissionError(f"No access to {path} for read operations")
    
    return True

def image_path_to_xml_path(image_path: Path, check: bool=True) -> Path:
    """
    Return the corresponding xml path for a image

    Args:
        image_path (Path): image path

    Raises:
        FileNotFoundError: no xml for image path
        PermissionError: xml file is not readable

    Returns:
        Path: xml path
    """
    xml_path = image_path.absolute().parent.joinpath("page", image_path.stem + '.xml')
    
    if check:
        check_path_accessible(xml_path)

    return xml_path

def xml_path_to_image_path(xml_path: Path, check: bool=True) -> Path:
    """
    Return the corresponding image path for an xml

    Args:
        xml_path (Path): xml path

    Raises:
        FileNotFoundError: no image for xml path
        PermissionError: image file in not readable

    Returns:
        Path: image_path
    """
    # IDEA add more image formats, maybe use glob to get the corresponding file
    # image_formats = [".bmp", ".dib",
    #                  ".jpeg", ".jpg", ".jpe",
    #                  ".jp2",
    #                  ".png",
    #                  ".webp",
    #                  ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
    #                  ".pfm",
    #                  ".sr", ".ras",
    #                  ".tiff", ".tif",
    #                  ".exr",
    #                  ".hdr", ".pic"]
    image_path = xml_path.absolute().parents[1].joinpath(xml_path.stem + ".jpg")
    
    if check:
        check_path_accessible(image_path)
        
    return image_path

def unique_path(path: str|Path, current_count: int=1) -> Path:
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

def clean_input(input_list: list[str], suffixes: Iterable[str]) -> list[Path]:
    """
    Return only the input paths that exist with a specified suffix

    Args:
        input_list (list[str]): a list of inputs, either filenames or a len==1 folder name
        suffixes (Iterable[str]): valid suffixes to check if the file is valid

    Raises:
        ValueError: No input set for the input_list
        FileNotFoundError: No files found with the suffixes

    Returns:
        list[Path]: cleaned Path variables
    """
    if len(input_list) == 0:
        raise ValueError("Must set the input")
    path_list: list[Path] = [Path(path) for path in input_list]
    
    for path in path_list:
        check_path_accessible(path)
    
    if len(path_list) == 1 and path_list[0].is_dir():
        path_list = [path for path in path_list[0].glob("*")]
    
    path_list = os_sorted([path for path in path_list if path.is_file() and path.suffix in suffixes])
    
    if len(path_list) == 0:
        raise FileNotFoundError("No valid files found in input")
    
    return path_list