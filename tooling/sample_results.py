import argparse
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy_utils import copy_mode
from utils.input_utils import is_path_supported_format, supported_image_formats
from utils.path_utils import image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copying sampled files from a large pagexml corpus to a different folder")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder", required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)

    parser.add_argument("-m", "--mode", choices=["link", "symlink", "copy"], help="Mode for moving the images", default="copy")

    k_value = parser.add_mutually_exclusive_group(required=True)
    k_value.add_argument("-p", "--percentage", help="Percentage of files to copy", type=float, default=None)
    k_value.add_argument("-n", "--number", help="Number of files to copy", type=int, default=None)

    args = parser.parse_args()
    return args


def copy_paths(paths: list[Path], output_dir, mode="copy") -> list[Path]:
    """
    copy a list of image paths to an output dir. The respective pageXMLs are also copied

    Args:
        paths (list[Path]): image paths
        output_dir (Path): path of the output dir
        mode (str, optional): type of copy mode (symlink, link, copy). Defaults to "copy".

    Returns:
        list[Path]: output paths
    """
    if not output_dir.is_dir():
        print(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)

    page_dir = output_dir.joinpath("page")

    if not page_dir.is_dir():
        print(f"Could not find output page dir ({page_dir}), creating one at specified location")
        page_dir.mkdir(parents=True)

    output_paths = []
    for path in paths:
        page_path = image_path_to_xml_path(path)
        output_path = output_dir.joinpath(path.name)
        output_page_path = page_dir.joinpath(page_path.name)
        copy_mode(path, output_path, mode=mode)
        copy_mode(page_path, output_page_path, mode=mode)

        output_paths.append(output_path)

    return output_paths


def main(args):
    """
    Sample result (or really an pageXML) from a larger corpus to process or view.

    Args:
        args (argparse.Namespace): arguments for where to find the images, and the output dir

    Raises:
        ValueError: must give an input
        ValueError: must give an output
        FileNotFoundError: input dir is missing
        FileNotFoundError: no images found in the input dir
        NotImplementedError: both the percentage and the number of samples is None
    """
    if args.input == "":
        raise ValueError("Must give an input")
    if args.output == "":
        raise ValueError("Must give an output")

    input_dir = Path(args.input)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"{input_dir} does not exist")

    input_dir_paths = input_dir.glob(f"*")
    all_image_paths = [path for path in input_dir_paths if is_path_supported_format(path, supported_image_formats)]

    if len(all_image_paths) == 0:
        raise FileNotFoundError(f"No images found within {input_dir}")

    if args.percentage is not None:
        assert 0 < args.percentage <= 1, f"Sample percentage is outside range (0-1], percentage: {args.percentage}"
        k_value = round(len(all_image_paths) * args.percentage)
    elif args.number is not None:
        k_value = args.number
    else:
        raise NotImplementedError

    assert (
        0 < k_value <= len(all_image_paths)
    ), f"Number of samples is outside range (0-{len(all_image_paths)}], number of samples: {k_value}"

    sampled_paths = random.sample(all_image_paths, k=k_value)

    output_dir = Path(args.output)

    if not output_dir.is_dir():
        print(f"Could not find output dir ({output_dir}), creating one at specified location")
        output_dir.mkdir(parents=True)

    output_paths = copy_paths(sampled_paths, output_dir, mode=args.mode)

    print(f"Copied {len(output_paths)} file to output path: {output_dir}")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
