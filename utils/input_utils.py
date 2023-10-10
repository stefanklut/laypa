from typing import Sequence
from pathlib import Path
import os

def clean_input_paths(input_paths: str | Path | Sequence[str|Path]) -> list[Path]:
    """
    Make all types of input path conform to list of paths
    
    Args:
        input_paths (str | Path | Sequence[str | Path]): path(s) to dir/file(s) with the location of paths

    Raises:
        ValueError: Must provide input path
        NotImplementedError: given input paths are the wrong class

    Returns:
        list[Path]: output paths of images
    """
    if not input_paths:
        raise ValueError("Must provide input path")
    
    if isinstance(input_paths, str):
        output = [Path(input_paths)]
    elif isinstance(input_paths, Path):
        output = [input_paths]
    elif isinstance(input_paths, Sequence):
        output = []
        for path in input_paths:
            if isinstance(path, str):
                output.append(Path(path))
            elif isinstance(path, Path):
                output.append(path)
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError
    
    return output

def get_file_paths(input_paths: str | Path | Sequence[str|Path], 
                   formats: Sequence[str], 
                   disable_check: bool=False) -> list[Path]:
    """
    Takes input paths, that may point to txt files containing more input paths and extracts them

    Args:
        input_paths (str | Path | Sequence[str | Path]): input path that have not been formatted
        formats (Sequence[str]): list of accepted file formats (extensions)
        disable_check (bool, optional): Run a check to see if all extracted files exist. Defaults to False.

    Raises:
        TypeError: input_paths is not set
        TypeError: formats are not set
        ValueError: formats are empty
        FileNotFoundError: input path not found on the filesystem
        PermissionError: input path not accessible
        FileNotFoundError: dir does not contain any files with the specified formats
        FileNotFoundError: file from txt file does not exist
        ValueError: specified path is not a dir or txt file

    Returns:
        list[Path]: output paths
    """
    if input_paths is None:
        raise TypeError("Cannot run when the input path is None")
    
    if formats is None:
        raise TypeError("Cannot run when the formats is None")
    
    if len(formats) == 0:
        raise ValueError("Must provide the accepted image types")
    
    input_paths = clean_input_paths(input_paths)
        
    output_paths = []
    
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input dir/file ({input_path}) is not found")
        
        if not os.access(path=input_path, mode=os.R_OK):
            raise PermissionError(
                f"No access to {input_path} for read operations")
        
        # IDEA This could be replaces with input_path.rglob(f"**/page/*.xml"), con: this remove the supported format check
        if input_path.is_dir():
            sub_output_paths = [image_path.absolute() for image_path in input_path.glob("*") if image_path.suffix in formats]
            
            if not disable_check:
                if len(sub_output_paths) == 0:
                    raise FileNotFoundError(f"No files found in the provided dir(s)/file(s) {input_path}")
                
        elif input_path.is_file() and input_path.suffix == ".txt":
            with input_path.open(mode="r") as f:
                paths_from_file = [Path(line) for line in f.read().splitlines()]
            sub_output_paths = [path if path.is_absolute() else input_path.parent.joinpath(path) for path in paths_from_file if path.suffix in formats]
            
            if len(sub_output_paths) == 0:
                    raise FileNotFoundError(f"No files found in the provided dir(s)/file(s) {input_path}")
            
            if not disable_check:
                for path in sub_output_paths:
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing file ({path}) from the txt file: {input_path}")
                        
        else:
            raise ValueError(f"Invalid file type: {input_path.suffix}")

        output_paths.extend(sub_output_paths)
    
    return output_paths