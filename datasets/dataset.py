from multiprocessing import Pool
import os
from typing import Optional

import numpy as np
import argparse
import ast

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

# IDEA Add the baseline generation and regions in the dataloader so they can scale with the images

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loading the image dataset to dict form")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)

    args = parser.parse_args()
    return args


def create_data(input_data: tuple[Path, Path, np.ndarray]) -> dict:
    """_summary_

    Args:
        input_data (tuple[Path, Path, np.ndarray]): input consisting of the image path, the labels mask path, and the shape of the image (use for grouping for example)

    Raises:
        FileNotFoundError: image path is missing
        FileNotFoundError: mask path is missing
        ValueError: the image and mask path have a different stem

    Returns:
        dict: data used for detectron training
    """
    image_path, mask_path, output_size = input_data

    # Data existence check
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path missing ({image_path})")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask path missing ({mask_path})")

    # Data_ids check
    if image_path.stem != mask_path.stem:
        raise ValueError(
            f"Image id should match mask id ({image_path.stem} vs {mask_path.stem}")

    # IDEA Include the pageXML file and get the segmentation for them for regions, maybe even baselines (for instance prediction)

    # objects = [["bbox": list[float],
    #            "bbox_mode": int,
    #            "category_id": int,
    #            "segmentation": list[list[float]],
    #            "keypoints": list[float],
    #            "iscrowd": 0 or 1,
    #            ] for anno in pagexml]

    # panos = [{"id": int,
    #           "category_id": int,
    #           "iscrowd": 0 or 1} for pano in pagexml
    #          ]

    data = {"file_name": str(image_path),
            "height": output_size[0],
            "width": output_size[1],
            "image_id": image_path.stem,
            # "annotations": objects
            "sem_seg_file_name": str(mask_path),
            # "pan_seg_file_name": str,
            # "segments_info": panos
            }
    return data


def dataset_dict_loader(dataset_dir: str | Path) -> list[dict]:
    """_summary_

    Args:
        dataset_dir (str | Path): dir containing three files,
            image_list.txt: containing the locations of the images, 
            mask_list.txt: containing the locations of the mask, 
            output_sizes.txt: containing the image shapes

    Raises:
        FileNotFoundError: image_list.txt does not exist
        FileNotFoundError: mask_list.txt does not exist
        FileNotFoundError: output_sizes.txt
        ValueError: length of the three list does not match

    Returns:
        list[dict]: list of dicts used to feed the dataloader
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    image_list = dataset_dir.joinpath("image_list.txt")
    if not image_list.is_file():
        raise FileNotFoundError(f"Image list is missing ({image_list})")

    mask_list = dataset_dir.joinpath("mask_list.txt")
    if not mask_list.is_file():
        raise FileNotFoundError(f"Mask list is missing ({mask_list})")

    output_sizes_list = dataset_dir.joinpath("output_sizes.txt")
    if not output_sizes_list.is_file():
        raise FileNotFoundError(
            f"Output sizes is missing ({output_sizes_list})")

    with open(image_list, mode='r') as f:
        image_paths = [dataset_dir.joinpath(line.strip()) for line in f.readlines()]

    with open(mask_list, mode='r') as f:
        mask_paths = [dataset_dir.joinpath(line.strip()) for line in f.readlines()]

    with open(output_sizes_list, mode='r') as f:
        output_sizes = f.readlines()
        output_sizes = [ast.literal_eval(output_size)
                        for output_size in output_sizes]

    # Data formatting check
    if not (len(image_paths) == len(mask_paths) == len(output_sizes)):
        raise ValueError(
            f"expecting the images, mask and output_sizes to be the same length: {len(image_paths)}, {len(mask_paths)}, {len(output_sizes)}")

    # Single Thread
    # input_dicts = []
    # for image_path, mask_path, output_size in zip(image_paths, mask_paths, output_sizes):
    #     input_dicts.append(create_data((image_path, mask_path, output_size)))

    # Multi Thread
    with Pool(os.cpu_count()) as pool:
        input_dicts = list(pool.imap_unordered(
            create_data, zip(image_paths, mask_paths, output_sizes)))

    return input_dicts

# IDEA register dataset for inference as well
def register_baseline(train: Optional[str|Path]=None, 
                      val: Optional[str|Path]=None, 
                      train_name: Optional[str]=None, 
                      val_name: Optional[str]=None,
                      ignore_label: int=255):
    """
    Register the baseline type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "baseline"],
            stuff_colors=[(0,0,0), (255,255,255)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "baseline"],
            stuff_colors=[(0,0,0), (255,255,255)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata

def register_region(train: Optional[str|Path]=None, 
                    val: Optional[str|Path]=None, 
                    train_name: Optional[str]=None, 
                    val_name: Optional[str]=None,
                    ignore_label: int=255):
    """
    Register the region type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"],
            stuff_colors=[(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"],
            stuff_colors=[(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata
        
def register_start(train: Optional[str|Path]=None, 
                   val: Optional[str|Path]=None, 
                   train_name: Optional[str]=None, 
                   val_name: Optional[str]=None,
                   ignore_label: int=255):
    """
    Register the start type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "start"],
            stuff_colors=[(0,0,0), (0,255,0)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "start"],
            stuff_colors=[(0,0,0), (0,255,0)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata

def register_end(train: Optional[str|Path]=None, 
                 val: Optional[str|Path]=None, 
                 train_name: Optional[str]=None, 
                 val_name: Optional[str]=None,
                 ignore_label: int=255):
    """
    Register the end type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "end"],
            stuff_colors=[(0,0,0), (255,0,0)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "end"],
            stuff_colors=[(0,0,0), (255,0,0)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata

def register_separator(train: Optional[str|Path]=None, 
                 val: Optional[str|Path]=None, 
                 train_name: Optional[str]=None, 
                 val_name: Optional[str]=None,
                 ignore_label: int=255):
    """
    Register the separator type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "separator"],
            stuff_colors=[(0,0,0), (50,50,50)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "separator"],
            stuff_colors=[(0,0,0), (50,50,50)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata

def register_baseline_separator(train: Optional[str|Path]=None, 
                 val: Optional[str|Path]=None, 
                 train_name: Optional[str]=None, 
                 val_name: Optional[str]=None,
                 ignore_label: int=255):
    """
    Register the baseline separator type dataset. Should be called by register_dataset function
    """
    metadata = None
    if train is not None and train != "":
        DatasetCatalog.register(
            name=train_name,
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get(train_name).set(
            stuff_classes=["background", "baseline", "separator"],
            stuff_colors=[(0,0,0), (255,255,255), (50,50,50)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            stuff_classes=["background", "baseline", "separator"],
            stuff_colors=[(0,0,0), (255,255,255), (50,50,50)],
            evaluator_type="sem_seg",
            ignore_label=ignore_label
        )
        metadata = MetadataCatalog.get(val_name)
    assert metadata is not None, "Metadata has not been set"
    return metadata

def register_dataset(train: Optional[str|Path]=None, 
                     val: Optional[str|Path]=None, 
                     train_name: Optional[str]=None, 
                     val_name: Optional[str]=None,
                     mode: Optional[str]=None,
                     ignore_label: int=255):
    """
    Register a dataset to the detectron dataset catalog

    Args:
        train (Optional[str | Path], optional): dir containing the txt files for the training. Defaults to None.
        val (Optional[str | Path], optional): dir containing the txt files for the validation. Defaults to None.
        train_name (Optional[str], optional): name under which the training dataset is saved in the catalog. Defaults to None.
        val_name (Optional[str], optional): name under which the validation dataset is saved in the catalog. Defaults to None.
        mode (Optional[str], optional): type of dataset being used. Defaults to None.
        ignore_label (int, optional): value in data that should be ignore (saved in metadata). Defaults to 255.

    Raises:
        NotImplementedError: mode is not an accepted value

    Returns:
        MetadataCatalog: metadata catalog for the registered dataset
    """
    assert train is not None or val is not None, "Must set at least something when registering"
    assert train is None or train_name is not None, "If train is not None, then train_name has to be set"
    assert val is None or val_name is not None, "If val is not None, then val_name has to be set"
    
    if mode == "baseline":
        return register_baseline(train, val, train_name, val_name, ignore_label=ignore_label)
    elif mode == "region":
        return register_region(train, val, train_name, val_name, ignore_label=ignore_label)
    elif mode == "start":
        return register_start(train, val, train_name, val_name, ignore_label=ignore_label)
    elif mode == "end":
        return register_end(train, val, train_name, val_name, ignore_label=ignore_label)
    elif mode == "separator":
        return register_separator(train, val, train_name, val_name, ignore_label=ignore_label)
    elif mode == "baseline_separator":
        return register_baseline_separator(train, val, train_name, val_name, ignore_label=ignore_label)
    else:
        raise NotImplementedError(
            f"Only have \"baseline\", \"start\", \"end\" and \"region\", given {mode}")

def main(args) -> None:
    results = dataset_dict_loader(args.input)
    print(results)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
