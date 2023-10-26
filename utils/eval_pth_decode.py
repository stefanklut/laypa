import argparse
import sys
from pathlib import Path

import torch

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
    Decode the metrics as saved in the pickle format. Not really used, mainly for seeing what the format is for
    """
    pth_path = Path(args.input)
    assert pth_path.suffix == ".pth", pth_path.suffix
    check_path_accessible(pth_path)

    metrics = torch.load(pth_path)
    print(metrics)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
