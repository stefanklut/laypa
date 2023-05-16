import os
from pathlib import Path
import re

def check_path_accessible(path: Path):
    """
    Check if the provide path is accessible, raise error for different checks
    Args:
        path (Path): path to check
    Raises:
        TypeError: path is not a Path object
        FileNotFoundError: folder/file does not exist at location
        PermissionError: no read access for folder/file
    """
    if not isinstance(path, Path):
        raise TypeError(f"provided object {path} is not Path, but {type(path)}")
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

    Returns:
        Path: image_path
    """
    
    
    image_formats = [".bmp", ".dib",
                     ".jpeg", ".jpg", ".jpe",
                     ".jp2",
                     ".png",
                     ".webp",
                     ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
                     ".pfm",
                     ".sr", ".ras",
                     ".tiff", ".tif",
                     ".exr",
                     ".hdr", ".pic"]
    
    # image_formats = [".jpg"]
    
    # Check if image with name exist if so return, else raise Error
    for image_format in image_formats:
        image_path = xml_path.absolute().parents[1].joinpath(xml_path.stem + image_format)
        if image_path.exists():
            break
    else:
        raise FileNotFoundError(f"No image equivalent found for {xml_path}")
    
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