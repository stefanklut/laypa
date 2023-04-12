import json
from multiprocessing import Pool
import os
from typing import Optional

import numpy as np
import argparse
import ast

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

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
        ValueError: the image and mask path have a different stem

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
        "file_name"        : str(image_path),
        "height"           : output_size[0],
        "width"            : output_size[1],
        "image_id"         : image_path.stem,
        "annotations"      : annotations,
        "sem_seg_file_name": str(mask_path),
        "pan_seg_file_name": str(pano_path),
        "segments_info"    : segments_info
    }
    return data


def dataset_dict_loader(dataset_dir: str | Path) -> list[dict]:
    """
    Create the dicts used for loading during training

    Args:
        dataset_dir (str | Path): dir containing the dataset

    Raises:

        ValueError: length of the info does not match

    Returns:
        list[dict]: list of dicts used to feed the dataloader
    """
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)
        
    info_path = dataset_dir.joinpath("info.json")
        
    with open(info_path, 'r') as f:
        input_data = json.load(f)

    # Unpack from dict of list -> list of dicts, add dataset dir to all str (assume they are paths)
    input_data = [{key: (dataset_dir.joinpath(value) if isinstance(value, str) else value) 
                    for (key, value) in zip(input_data.keys(), t)} for t in zip(*input_data.values())]

    # Single Thread
    # input_dicts = []
    # for input_data_i in input_data:
    #     input_dicts.append(create_data(input_data_i))

    # Multi Thread
    with Pool(os.cpu_count()) as pool:
        input_dicts = list(pool.imap_unordered(
            create_data, input_data))

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
            thing_classes  = ["baseline"],
            thing_colors   = [(255,255,255)],
            stuff_classes  = ["background", "baseline"],
            stuff_colors   = [(0,0,0), (255,255,255)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            thing_classes  = ["baseline"],
            thing_colors   = [(255,255,255)],
            stuff_classes  = ["background", "baseline"],
            stuff_colors   = [(0,0,0), (255,255,255)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
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
        # HACK This only works for one type of stuff/thing classes
        MetadataCatalog.get(train_name).set(
            # thing_classes  = ["marginalia", "page-number", "resolution", "date", "index", "attendance"],
            # thing_colors   = [(228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            # stuff_classes  = ["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"],
            # stuff_colors   = [(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            thing_classes  = ["paragraph", "signature-mark", "catch-word", "page-number",
              "header", "heading", "marginalia"],
            thing_colors   = [(228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135), (0,140,255)],
            stuff_classes  = ["background", "paragraph", "signature-mark", "catch-word", "page-number",
              "header", "heading", "marginalia"],
            stuff_colors   = [(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135), (0,140,255)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            # thing_classes  = ["marginalia", "page-number", "resolution", "date", "index", "attendance"],
            # thing_colors   = [(228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            # stuff_classes  = ["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"],
            # stuff_colors   = [(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135)],
            thing_classes  = ["paragraph", "signature-mark", "catch-word", "page-number",
              "header", "heading", "marginalia"],
            thing_colors   = [(228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135), (0,140,255)],
            stuff_classes  = ["background", "paragraph", "signature-mark", "catch-word", "page-number",
              "header", "heading", "marginalia"],
            stuff_colors   = [(0,0,0), (228,3,3), (255,140,0), (255,237,0), (0,128,38), (0,77,255), (117,7,135), (0,140,255)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
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
            thing_classes  = ["start"],
            thing_colors   = [(0,255,0)],
            stuff_classes  = ["background", "start"],
            stuff_colors   = [(0,0,0), (0,255,0)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            thing_classes  = ["start"],
            thing_colors   = [(0,255,0)],
            stuff_classes  = ["background", "start"],
            stuff_colors   = [(0,0,0), (0,255,0)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
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
            thing_classes=["end"],
            thing_colors=[(255,0,0)],
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
            thing_classes=["end"],
            thing_colors=[(255,0,0)],
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
            thing_classes  = ["separator"],
            thing_colors   = [(50,50,50)],
            stuff_classes  = ["background", "separator"],
            stuff_colors   = [(0,0,0), (50,50,50)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            thing_classes  = ["separator"],
            thing_colors   = [(50,50,50)],
            stuff_classes  = ["background", "separator"],
            stuff_colors   = [(0,0,0), (50,50,50)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
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
            thing_classes  = ["baseline", "separator"],
            thing_colors   = [(255,255,255), (50,50,50)],
            stuff_classes  = ["background", "baseline", "separator"],
            stuff_colors   = [(0,0,0), (255,255,255), (50,50,50)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
        )
        metadata = MetadataCatalog.get(train_name)
    if val is not None and val != "":
        DatasetCatalog.register(
            name=val_name,
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get(val_name).set(
            thing_classes  = ["baseline", "separator"],
            thing_colors   = [(255,255,255), (50,50,50)],
            stuff_classes  = ["background", "baseline", "separator"],
            stuff_colors   = [(0,0,0), (255,255,255), (50,50,50)],
            evaluator_type = "sem_seg",
            ignore_label   = ignore_label
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
    
    mapping = {
        "baseline"          : register_baseline,
        "region"            : register_region,
        "start"             : register_start,
        "end"               : register_end,
        "separator"         : register_separator,
        "baseline_separator": register_baseline_separator,
    }
    
    if mode in mapping:
        return mapping[mode](train, val, train_name, val_name, ignore_label=ignore_label)
    else:
        raise NotImplementedError(
            f"Only have {mapping.keys()}, given {mode}")

def main(args):
    results = dataset_dict_loader(args.input)
    return results


if __name__ == "__main__":
    args = get_arguments()
    results = main(args)
    print(results[0])
    from detectron2.utils.visualizer import Visualizer
    import cv2
    import matplotlib.pyplot as plt
    metadata = register_dataset(train=args.input, train_name="train", mode="region")
    for result in results:
        image = cv2.imread(result["file_name"])
        sem_seg = cv2.imread(result["sem_seg_file_name"])
        pano = cv2.imread(result["pan_seg_file_name"], cv2.IMREAD_COLOR)
        visualizer = Visualizer(image[..., ::-1].copy(), metadata=metadata, scale=1)
        # vis = visualizer.overlay_instances(
        #     boxes=[result['bbox'] for result in result["annotations"]],
        #     labels=[result['category_id'] for result in result["annotations"]],
        #     masks=[result['segmentation'] for result in result["annotations"]],
        # )
        vis = visualizer.draw_dataset_dict(result)
        # DOES NOT WORK FOR SOME REASON
        # vis = visualizer.draw_panoptic_seg(panoptic_seg=pano, segments_info=result["segments_info"], area_threshold=0, alpha=0.5)
        plt.imshow(vis.get_image())
        plt.show()