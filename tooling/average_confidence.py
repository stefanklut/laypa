import argparse
import sys
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xmlPAGE import PageData
from utils.input_utils import get_file_paths


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation of model compared to ground truth")

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    args = parser.parse_args()
    return args


def get_confidence_from_pagexml(path: Path):
    page_data = PageData(path)
    page_data.parse()

    metadata_items = page_data._iter_element("MetadataItem")
    for metadata_item in metadata_items:
        if metadata_item.attrib["value"] != "laypa":
            continue

        labels = page_data._iter_subelement(metadata_item, "Label")
        for label in labels:
            if label.attrib["type"] != "confidence":
                continue

            return float(label.attrib["value"])

    return None


def main(args):
    xml_paths = get_file_paths(args.input, [".xml"])

    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(get_confidence_from_pagexml, xml_paths), total=len(xml_paths)))

    if any([result is None for result in results]):
        raise ValueError("Some confidence values are missing")

    results = np.asarray(results)
    average_confidence = np.mean(results)
    std_confidence = np.std(results)

    print(f"Average confidence: {average_confidence}, Standard deviation: {std_confidence}")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
