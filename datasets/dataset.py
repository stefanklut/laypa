import os

import numpy as np
import argparse
import ast

from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog

# IDEA Add the baseline generation and regions in the dataloader so they can scale with the images

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    parser.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)

    args = parser.parse_args()
    return args


def create_data(input_data):
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


def dataset_dict_loader(dataset_dir: str | Path):
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
        image_paths = [Path(line.strip()) for line in f.readlines()]

    with open(mask_list, mode='r') as f:
        mask_paths = [Path(line.strip()) for line in f.readlines()]

    with open(output_sizes_list, mode='r') as f:
        output_sizes = f.readlines()
        output_sizes = [ast.literal_eval(output_size)
                        for output_size in output_sizes]

    # Data formatting check
    if not (len(image_paths) == len(mask_paths) == len(output_sizes)):
        raise ValueError(
            "expecting the images, mask and output_sizes to be the same lenght")

    # Single Thread
    input_dicts = []
    for image_path, mask_path, output_size in zip(image_paths, mask_paths, output_sizes):
        input_dicts.append(create_data((image_path, mask_path, output_size)))

    # TODO Multi Thread?

    return input_dicts

# IDEA register dataset for inference aswell
def register_baseline(train=None, val=None):
    if train is not None and train != "":
        DatasetCatalog.register(
            name="pagexml_baseline_train",
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get("pagexml_baseline_train").set(stuff_classes=["backgroud", "baseline"])
        MetadataCatalog.get("pagexml_baseline_train").set(stuff_colors=[(0,0,0), (255,255,255)])
        MetadataCatalog.get("pagexml_baseline_train").set(evaluator_type="sem_seg")
        MetadataCatalog.get("pagexml_baseline_train").set(ignore_label=255)
    if val is not None and val != "":
        DatasetCatalog.register(
            name="pagexml_baseline_val",
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get("pagexml_baseline_val").set(stuff_classes=["backgroud", "baseline"])
        MetadataCatalog.get("pagexml_baseline_val").set(stuff_colors=[(0,0,0), (255,255,255)])
        MetadataCatalog.get("pagexml_baseline_val").set(evaluator_type="sem_seg")
        MetadataCatalog.get("pagexml_baseline_val").set(ignore_label=255)

def register_region(train=None, val=None):
    if train is not None and train != "":
        DatasetCatalog.register(
            name="pagexml_region_train",
            func=lambda path=train: dataset_dict_loader(path)
        )
        MetadataCatalog.get("pagexml_region_train").set(stuff_classes=["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"])
        MetadataCatalog.get("pagexml_region_train").set(stuff_colors=[(0,0,0), (3,3,228), (0,140,255), (0,237,255), (38,128,0), (255,77,0), (135,7,117)])
        MetadataCatalog.get("pagexml_region_train").set(evaluator_type="sem_seg")
        MetadataCatalog.get("pagexml_region_train").set(ignore_label=255)
    if val is not None and val != "":
        DatasetCatalog.register(
            name="pagexml_region_val",
            func=lambda path=val: dataset_dict_loader(path)
        )
        MetadataCatalog.get("pagexml_region_val").set(stuff_classes=["background", "marginalia", "page-number", "resolution", "date", "index", "attendance"])
        MetadataCatalog.get("pagexml_region_val").set(stuff_colors=[(0,0,0), (3,3,228), (0,140,255), (0,237,255), (38,128,0), (255,77,0), (135,7,117)])
        MetadataCatalog.get("pagexml_region_val").set(evaluator_type="sem_seg")
        MetadataCatalog.get("pagexml_region_val").set(ignore_label=255)

def main(args) -> None:
    results = dataset_dict_loader(args.input)
    print(results)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
