import argparse
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from natsort import os_sorted

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode
from utils.image_utils import load_image_array_from_path
from utils.logging_utils import get_logger_name
from utils.path_utils import image_path_to_xml_path, xml_path_to_image_path
from utils.regions_from_dataset import count_regions_single_page

logger = logging.getLogger(get_logger_name())


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copying files from multiple folders into a single structured dataset")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="input folders", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)

    parser.add_argument("-c", "--copy", action="store_true", help="Copy the images to the output folder location")
    parser.add_argument("-m", "--mode", choices=["link", "symlink", "copy"], help="Mode for moving the images", default="copy")
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="The percentages of each split (train,val,test), if the sum does not equal 1 relative percentage are taken(6,2,2 -> 0.6,0.2,0.2)",
    )
    parser.add_argument(
        "--check",
        nargs="?",
        const="all",
        default=None,
        help="Check if images is not corrupted, and if pageXML is properly formatted",
    )

    args = parser.parse_args()
    return args


def copy_xml_paths(xml_paths: list[Path], output_dir: Path, mode: str = "copy") -> list[Path]:
    """
    copy a list of pageXML paths to an output dir. The respective images are also copied

    Args:
        xml_paths (list[Path]): image paths
        output_dir (Path): path of the output dir
        mode (str, optional): type of copy mode (symlink, link, copy). Defaults to "copy".

    Returns:
        list[Path]: output paths
    """
    if not output_dir.is_dir():
        logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)

    page_dir = output_dir.joinpath("page")

    if not page_dir.is_dir():
        logger.info(f"Could not find output page dir ({page_dir}), creating one at specified location")
        page_dir.mkdir(parents=True)

    output_paths = []
    for xml_path in xml_paths:
        image_path = xml_path_to_image_path(xml_path)
        output_image_path = output_dir.joinpath(image_path.name)
        output_xml_path = page_dir.joinpath(xml_path.name)
        copy_mode(image_path, output_image_path, mode=mode)
        copy_mode(xml_path, output_xml_path, mode=mode)

        output_paths.append(output_xml_path)

    return output_paths


def copy_image_paths(image_paths: list[Path], output_dir: Path, mode: str = "copy") -> list[Path]:
    """
    copy a list of image paths to an output dir. The respective pageXMLs are also copied

    Args:
        image_paths (list[Path]): image paths
        output_dir (Path): path of the output dir
        mode (str, optional): type of copy mode (symlink, link, copy). Defaults to "copy".

    Returns:
        list[Path]: output paths
    """
    if not output_dir.is_dir():
        logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)

    page_dir = output_dir.joinpath("page")

    if not page_dir.is_dir():
        logger.info(f"Could not find output page dir ({page_dir}), creating one at specified location")
        page_dir.mkdir(parents=True)

    output_paths = []
    for image_path in image_paths:
        xml_path = image_path_to_xml_path(image_path)
        output_image_path = output_dir.joinpath(image_path.name)
        output_xml_path = page_dir.joinpath(xml_path.name)
        copy_mode(image_path, output_image_path, mode=mode)
        copy_mode(xml_path, output_xml_path, mode=mode)

        output_paths.append(output_image_path)

    return output_paths


def train_test_split(data: list, train_size: float):
    """
    Splits the given data into training and testing sets based on the specified train size.

    Args:
        data (list): The data to be split.
        train_size (float): The proportion of data to be used for training.

    Returns:
        tuple: A tuple containing the training data and testing data.
    """

    assert train_size >= 0 and train_size <= 1, "Train size must be between 0 and 1"

    np.random.shuffle(data)
    train_len = int(len(data) * train_size)
    train_data = data[:train_len]
    test_data = data[train_len:]
    return train_data, test_data


def main(args):
    """
    Create a dataset structure (train, val, test) from multiple sub dirs

    Args:
        args (argparse.Namespace): arguments for where to find the images, and the output dir

    Raises:
        ValueError: must give an input
        ValueError: must give an output
        FileNotFoundError: input dir is missing
        ValueError: found duplicates in the images
        FileNotFoundError: no images found in sub dir
    """

    if args.input == []:
        raise ValueError("Must give an input")
    if args.output == "":
        raise ValueError("Must give an output")

    input_paths = [Path(path) for path in args.input]

    dir_image_paths = []
    txt_image_paths = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} does not exist")

        if input_path.is_dir():
            # Get all pageXMLs somewhere in the folders
            xml_paths = list(input_path.rglob(f"**/page/*.xml"))
            if len(xml_paths) == 0:
                raise FileNotFoundError(f"No xml_files found within {input_path}")
            dir_image_paths.extend(xml_paths)
        elif input_path.is_file and input_path.suffix == ".txt":
            with input_path.open(mode="r") as f:
                paths_from_file = [Path(line) for line in f.read().splitlines()]
            txt_image_paths.extend(
                [path if path.is_absolute() else input_path.parent.joinpath(path) for path in paths_from_file]
            )

            for image_path in txt_image_paths:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing file from the txt file: {input_path}")

        else:
            raise ValueError(f"Invalid file type: {input_path.suffix}")

    dir_image_paths = [xml_path_to_image_path(path).absolute() for path in dir_image_paths]
    txt_image_paths = [path.absolute() for path in txt_image_paths]
    all_image_paths = os_sorted(dir_image_paths + txt_image_paths, key=str)

    if args.check is not None:
        if args.check not in ["all", "image", "page"]:
            raise ValueError(f"{args.save} is not a valid check mode")
        temp_image_paths = []
        for image_path in all_image_paths:
            image = load_image_array_from_path(image_path)
            if args.check in ["all", "image"]:
                if image is None:
                    continue
            if args.check in ["all", "page"]:
                xml_path = image_path_to_xml_path(image_path)
                counter = count_regions_single_page(xml_path)
                if None in counter.keys():
                    continue

            temp_image_paths.append(image_path)
        all_image_paths = temp_image_paths

    if len(all_image_paths) != len(set(path.stem for path in all_image_paths)):
        duplicates = {k: v for k, v in Counter(path.stem for path in all_image_paths).items() if v > 1}

        raise ValueError(f"Found duplicate stems for images\n {os_sorted(duplicates.items(), key=lambda s: str(s[0]))}")

    train_size, val_size, test_size = (split := np.asarray(args.split)) / np.sum(split)
    if train_size == 1:
        train_paths = all_image_paths
        val_paths = []
        test_paths = []
    elif val_size == 1:
        train_paths = []
        val_paths = all_image_paths
        test_paths = []
    elif test_size == 1:
        train_paths = []
        val_paths = []
        test_paths = all_image_paths
    else:
        train_paths, val_test_paths = train_test_split(all_image_paths, train_size=train_size)

        relative_val_size = val_size / (val_size + test_size)
        if relative_val_size == 1:
            val_paths = val_test_paths
            test_paths = []
        elif relative_val_size == 0:
            val_paths = []
            test_paths = val_test_paths
        else:
            val_paths, test_paths = train_test_split(val_test_paths, train_size=relative_val_size)

    logger.info("Number of train images:", len(train_paths))
    logger.info("Number of validation images:", len(val_paths))
    logger.info("Number of test images:", len(test_paths))

    output_dir = Path(args.output)

    if not output_dir.is_dir():
        logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)

    train_dir = output_dir.joinpath("train")
    val_dir = output_dir.joinpath("val")
    test_dir = output_dir.joinpath("test")

    if args.copy:
        train_paths = copy_image_paths(train_paths, train_dir, mode=args.mode)
        val_paths = copy_image_paths(val_paths, val_dir, mode=args.mode)
        test_paths = copy_image_paths(test_paths, test_dir, mode=args.mode)

    train_paths = [path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve() for path in train_paths]
    val_paths = [path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve() for path in val_paths]
    test_paths = [path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve() for path in test_paths]

    if train_paths:
        with output_dir.joinpath("train_filelist.txt").open(mode="w") as f:
            for train_path in train_paths:
                f.write(f"{train_path}\n")
    if val_paths:
        with output_dir.joinpath("val_filelist.txt").open(mode="w") as f:
            for val_path in val_paths:
                f.write(f"{val_path}\n")
    if test_paths:
        with output_dir.joinpath("test_filelist.txt").open(mode="w") as f:
            for test_path in test_paths:
                f.write(f"{test_path}\n")

    with output_dir.joinpath("info.txt").open(mode="w") as f:
        f.write(f"Created: {datetime.now()}\n")
        f.write(f"Number of train images: {len(train_paths)}\n")
        f.write(f"Number of validation images: {len(val_paths)}\n")
        f.write(f"Number of test images: {len(test_paths)}\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
