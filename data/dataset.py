import argparse
import json

# from multiprocessing.pool import ThreadPool as Pool
import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Optional

from detectron2.data import DatasetCatalog, Metadata, MetadataCatalog

from utils.color_utils import n_distinct_colors


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loading the image dataset to dict form")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder", required=True, type=str)

    args = parser.parse_args()
    return args


def create_data(input_data: dict) -> dict:
    """
    Return a single dict used for training

    Args:
        input_data (dict): input dict to be parsed

    Raises:
        FileNotFoundError: one of the given paths is missing

    Returns:
        dict: data used for detectron training
    """
    data = {}

    for key, value in input_data.items():
        path = Path(value)
        if path.is_file():
            if path.suffix == ".json":
                with path.open(mode="r") as f:
                    data[key] = json.load(f)
            else:
                data[key] = str(path)

                if key == "image_file_name":
                    data["file_name"] = str(path)
        else:
            raise FileNotFoundError(f"Path missing ({path})")

    return data


def dataset_dict_loader(input_data: list[dict]) -> list[dict]:
    """
    Create the dicts used for loading during training

    Args:
        input_data (list[dict]): list of dicts, each dict is one image and associated ground truth

    Returns:
        list[dict]: list of dicts used to feed the dataloader
    """
    # Single Thread
    # input_dicts = []
    # for input_data_i in input_data:
    #     input_dicts.append(create_data(input_data_i))

    # Multi Thread
    with Pool(os.cpu_count()) as pool:
        input_dicts = list(pool.imap_unordered(create_data, input_data))

    return input_dicts


def dict_of_list_to_list_of_dicts(input_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Convert a dict of lists into a list of dicts. All list much have the same length. The output number of dicts matches the length of the list

    Args:
        input_dict (dict[str, list[Any]]): dict of lists

    Returns:
        list[dict[str, Any]]: list of dicts
    """
    output_list = [{key: value for key, value in zip(input_dict.keys(), t)} for t in zip(*input_dict.values())]
    return output_list


def convert_to_paths(dataset_dir: Path, input_data: dict[str, list]) -> list[dict[str, Path]]:
    """
    Turn expected paths into actual Path objects instead of just strings

    Args:
        dataset_dir (Path): base dataset dir
        input_data (dict[str, list]): data dict with lists of paths

    Returns:
        list[dict[str, Path]]: list of dicts containing paths
    """
    converted_data = dict_of_list_to_list_of_dicts(input_data)
    converted_data = [{key: dataset_dir.joinpath(value) for key, value in item.items()} for item in converted_data]
    return converted_data


def classes_to_colors(classes: list[str], grayscale=False) -> list[int] | list[tuple[int, int, int]]:
    """
    Assign a unique distinct color to each class

    Args:
        classes (list[str]): names of classes
        grayscale (bool, optional): flag indicating whether to use grayscale colors. Defaults to False.

    Raises:
        ValueError: must have at least two classes

    Returns:
        list[int] | list[tuple[int, int, int]]: list of colors
    """
    if len(classes) < 2:
        raise ValueError(f"Expecting at least 2 classes got {len(classes)}")
    if grayscale:
        background_color = 0
        line_color = 255
    else:
        background_color = (0, 0, 0)
        line_color = (255, 255, 255)

    if len(classes) == 2:
        return [background_color, line_color]  # type: ignore

    # No rng should give the same colors every time
    distinct_colors = n_distinct_colors(len(classes) - 1, grayscale=grayscale, rng=0)
    colors = [background_color] + distinct_colors
    return colors  # type: ignore


def metadata_from_classes(classes: list[str], ignore_label: int = 255) -> Metadata:
    """
    Create unique metadata based on the give class names

    Args:
        classes (list[str]): names of classes
        ignore_label (int, optional): ignored class label. Defaults to 255.

    Returns:
        Metadata: metadata object
    """

    colors = classes_to_colors(classes)
    metadata = Metadata(
        thing_classes=classes[1:],
        thing_colors=colors[1:],
        stuff_classes=classes,
        stuff_colors=colors,
        ignore_label=ignore_label,
    )
    return metadata


def register_dataset(path: str | Path, name: str, ignore_label: int = 255) -> Metadata:
    """
    Register a dataset that was created by the preprocessing

    Args:
        path (str | Path): path to data on disk
        name (str): name of registered dataset
        ignore_label (int, optional): ignored class label. Defaults to 255.

    Returns:
        Metadata: metadata object
    """
    if isinstance(path, str):
        path = Path(path)

    info_path = path.joinpath("info.json")

    with info_path.open(mode="r") as f:
        info = json.load(f)

    data = convert_to_paths(path, info["data"])
    classes = info["classes"]

    list_of_dicts = dataset_dict_loader(data)

    DatasetCatalog.register(name=name, func=lambda: list_of_dicts)

    MetadataCatalog[name] = metadata_from_classes(classes, ignore_label)
    return MetadataCatalog.get(name)


def register_datasets(
    train: Optional[str | Path] = None,
    val: Optional[str | Path] = None,
    train_name: Optional[str] = None,
    val_name: Optional[str] = None,
    ignore_label: int = 255,
):
    """
    Register train and/or validation dataset

    Args:
        train (Optional[str | Path], optional): path to train data on disk. Defaults to None.
        val (Optional[str | Path], optional): path to val data on disk. Defaults to None.
        train_name (Optional[str], optional): name of train dataset. Defaults to None.
        val_name (Optional[str], optional): name of val dataset. Defaults to None.
        ignore_label (int, optional): ignored class label. Defaults to 255.

    Returns:
        Metadata: metadata object
    """

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
