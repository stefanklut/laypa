import json
from multiprocessing import Pool
import os
from typing import Optional

import distinctipy
import argparse

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog, Metadata

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loading the image dataset to dict form")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)

    args = parser.parse_args()
    return args


def create_data(input_data: dict) -> dict:
    """
    Return a single dict used for training

    Args:
        input_data (dict): input consisting of the image path, 
            the labels mask path, the instances path, and the shape of the image (use for grouping for example)

    Raises:
        FileNotFoundError: image path is missing
        FileNotFoundError: mask path is missing

    Returns:
        dict: data used for detectron training
    """
    image_path = Path(input_data["image_paths"])
    mask_path = Path(input_data["sem_seg_paths"])
    instances_path = Path(input_data["instances_paths"])
    pano_path = Path(input_data["pano_paths"])
    segments_info_path = Path(input_data["segments_info_paths"])
    output_size = input_data["output_sizes"]

    # Data existence check
    if not image_path.is_file(): 
        raise FileNotFoundError(f"Image path missing ({image_path})")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask path missing ({mask_path})")
    if not instances_path.is_file():
        raise FileNotFoundError(f"Instance path missing ({instances_path})")
    if not pano_path.is_file():
        raise FileNotFoundError(f"Pano path missing ({pano_path})")
    if not segments_info_path.is_file():
        raise FileNotFoundError(f"Segments info path missing ({segments_info_path})")

    with open(instances_path, 'r') as f:
        annotations = json.load(f)["annotations"]
    
    with open(segments_info_path, 'r') as f:
        segments_info = json.load(f)["segments_info"]

    data = {
        "file_name"         : str(image_path),
        "original_file_name": str(input_data["original_image_paths"]),
        "height"            : output_size[0],
        "width"             : output_size[1],
        "image_id"          : image_path.stem,
        "annotations"       : annotations,
        "sem_seg_file_name" : str(mask_path),
        "pan_seg_file_name" : str(pano_path),
        "segments_info"     : segments_info
    }
    return data


def dataset_dict_loader(input_data: list[dict]) -> list[dict]:
    """
    Create the dicts used for loading during training

    Args:
        dataset_dir (str | Path): dir containing the dataset

    Raises:

        ValueError: length of the info does not match

    Returns:
        list[dict]: list of dicts used to feed the dataloader
    """
    # Single Thread
    # input_dicts = []
    # for input_data_i in input_data:
    #     input_dicts.append(create_data(input_data_i))

    # Multi Thread
    with Pool(os.cpu_count()) as pool:
        input_dicts = list(pool.imap_unordered(
            create_data, input_data))

    return input_dicts

def dict_of_list_to_list_of_dicts(input_dict: dict[str, list]):
    output_list = [{key: value for key, value in zip(input_dict.keys(), t)} 
                        for t in zip(*input_dict.values())]
    return output_list

def convert_to_paths(dataset_dir, input_data):
    input_data = dict_of_list_to_list_of_dicts(input_data)
    input_data = [{key: dataset_dir.joinpath(value) if "paths" in key else value 
                    for key, value in item.items()} 
                    for item in input_data]
    return input_data

def classes_to_colors(classes) -> list[tuple[int,int,int]]:
    if len(classes) < 2:
        raise ValueError(f"Expecting at least 2 classes got {len(classes)}")
    background_color = (0,0,0)
    line_color = (255,255,255)
    if len(classes) == 2:
        return [background_color, line_color]
    
    colors = [background_color]
    distinct_colors = distinctipy.get_colors(len(classes)-1, rng=0) #no rng should give the same colors
    colors = [background_color] + [tuple(map(lambda channel: int(channel*255),color)) for color in distinct_colors]
    return colors

def metadata_from_classes(classes: list[str],
                          ignore_label: int=255):
    colors = classes_to_colors(classes)
    metadata = Metadata(
        thing_classes  = classes[1:],
        thing_colors   = colors[1:],
        stuff_classes  = classes,
        stuff_colors   = colors,
        evaluator_type = "sem_seg",
        ignore_label   = ignore_label
    )
    return metadata

def register_dataset(path:str | Path, 
                   name: str, 
                   ignore_label: int=255):
    if isinstance(path, str):
        path = Path(path)
    
    info_path = path.joinpath("info.json")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    data = convert_to_paths(path, info["data"])
    classes = info["classes"]
    
    DatasetCatalog.register(
        name=name,
        func=lambda data=data: dataset_dict_loader(data)
    )
    
    MetadataCatalog[name] = metadata_from_classes(classes, ignore_label)
    return MetadataCatalog.get(name)

def register_datasets(train: Optional[str|Path]=None, 
                     val: Optional[str|Path]=None, 
                     train_name: Optional[str]=None, 
                     val_name: Optional[str]=None,
                     ignore_label: int=255):
    
    assert train is not None or val is not None, "Must set at least something when registering"
    assert train is None or train_name is not None, "If train is not None, then train_name has to be set"
    assert val is None or val_name is not None, "If val is not None, then val_name has to be set"
    
    metadata = None
    if train and train_name:
        metadata = register_dataset(train, train_name, ignore_label)
    if val and val_name:
        metadata = register_dataset(val, val_name, ignore_label)
    assert metadata is not None, "Metadata has not been set"
    return metadata