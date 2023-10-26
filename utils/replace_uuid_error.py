import argparse

# from multiprocessing.pool import ThreadPool as Pool
import os
import re
import uuid
from multiprocessing.pool import Pool
from pathlib import Path

from tqdm import tqdm


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace the errored uuid")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder", required=True, type=str)
    args = parser.parse_args()
    return args


def replace_error(file_path):
    def gen_uuid(match_obj):
        return str(uuid.uuid4())

    with file_path.open(mode="r") as file:
        data = str(file.read())

    data = re.sub(r"&lt;module 'uuid' from '/data/stefank/miniconda3/envs/laypa/lib/python3.10/uuid.py'>", gen_uuid, data)

    with file_path.open(mode="w") as file:
        file.write(data)


def main(args):
    """
    Use to fix an error with the UUID, currently no longer used. It might be useful again if something similar happens again
    """
    input_folder = Path(args.input)

    if not input_folder.is_dir():
        raise FileNotFoundError("Folder not found")

    file_paths = list(input_folder.glob("*.xml"))

    if len(file_paths) == 0:
        raise FileNotFoundError("No XML Found")

    # Single Thread
    # for file_path in tqdm(file_paths):
    #     replace_error(file_path)

    # Multi Thread
    with Pool(os.cpu_count()) as pool:
        _ = list(tqdm(iterable=pool.imap_unordered(replace_error, file_paths), total=len(file_paths), desc="Fixing Error"))


if __name__ == "__main__":
    args = get_arguments()
    main(args)
