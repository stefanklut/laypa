import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.path_utils import check_path_accessible


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="load an decode the json_predictions to arrays")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)
    # io_args.add_argument(
    #     "-o", "--output", help="Output file", required=True, type=str)

    args = parser.parse_args()
    return args


def main(args):
    """
    Decode the results as saved in the json coco format. Not really used, mainly for seeing what the format is for
    """
    json_path = Path(args.input)
    assert json_path.suffix == ".json", json_path.suffix
    check_path_accessible(json_path)

    with json_path.open(mode="r") as f:
        predictions = json.load(f)

    for prediction in predictions:
        original_path = Path(prediction["file_name"])
        category = prediction["category_id"]
        segmentation = np.asarray(mask_utils.decode(prediction["segmentation"]))
        print(category)
        plt.imshow(segmentation, cmap="gray")
        plt.show()
        break


if __name__ == "__main__":
    args = get_arguments()
    main(args)
