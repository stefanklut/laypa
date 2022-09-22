import os

import numpy as np
import argparse
import ast

from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    parser.add_argument("-i", "--input", help="Input folder",
                        required=True, type=str)

    args = parser.parse_args()
    return args

def create_data(paths):
    pass

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
        image_paths = [Path(line) for line in f.readlines()]

    with open(mask_list, mode='r') as f:
        mask_paths = [Path(line) for line in f.readlines()]

    with open(output_sizes_list, mode='r') as f:
        output_sizes = f.readlines()
        output_sizes = [ast.literal_eval(output_size)
                        for output_size in output_sizes]
        print(output_sizes)

    # Data formatting check
    if not (len(image_paths) == len(mask_paths) == len(output_sizes)):
        raise ValueError(
            "expecting the images, mask and output_sizes to be the same lenght")
    
    print(zip(image_paths, mask_paths))

    for image_path, mask_path in zip(image_paths, mask_paths):
        # Data existence check
        if not image_path.is_file():
            raise FileNotFoundError(f"Image path missing ({image_path})")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask path missing ({mask_path})")

        # Data_ids check
        if image_path.stem != mask_path.stem:
            raise ValueError(
                f"Image id should match mask id ({image_path.stem} vs {mask_path.stem}")


def main(args):
    results = dataset_dict_loader(args.input)
    print(results)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
