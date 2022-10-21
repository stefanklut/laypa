from pathlib import Path
from typing import Iterable, Optional
from natsort import os_sorted

def get_page_xml(path: Path):
    xml_path = path.parent.joinpath("page", path.stem + '.xml')
    if not xml_path.exists():
        raise FileNotFoundError(f"No {xml_path} found")
    if not xml_path.is_file():
        raise IsADirectoryError(f"{xml_path} should be a file not directory")
        
    return path.parent.joinpath("page", path.stem + '.xml')

def clean_input(input_list: list[str], suffixes: Iterable[str]):
    if len(input_list) == 0:
        raise ValueError("Must set the input")
    path_list: list[Path] = [Path(path) for path in input_list]
    
    if len(path_list) == 1 and path_list[0].is_dir():
        path_list = [path for path in path_list[0].glob("*")]
    
    path_list = os_sorted([path for path in path_list if path.is_file() and path.suffix in suffixes])
    
    if len(path_list) == 0:
        raise FileNotFoundError("No valid files found in input")
    
    return path_list