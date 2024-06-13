import argparse

# from multiprocessing.pool import ThreadPool as Pool
import os
import sys
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xmlPAGE import PageData
from utils.input_utils import get_file_paths, supported_image_formats
from utils.path_utils import image_path_to_xml_path
from xml_comparison import pretty_print


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count regions from a dataset")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)
    args = parser.parse_args()
    return args


def count_regions_single_page(xml_path: Path) -> Counter:
    """
    Count the unique regions in a pageXML

    Args:
        xml_path (Path): Path to pageXML

    Returns:
        Counter: Count of all unique regions
    """
    page_data = PageData(xml_path)
    page_data.parse()

    region_names = ["TextRegion"]  # Assuming this is all there is
    zones = page_data.get_zones(region_names)

    if zones is None:
        return Counter()

    counter = Counter(item["type"] for item in zones.values())
    return counter


def main(args):
    """
    Run the full count over all pageXMLs found in the input dir

    Args:
        args (argparse.Namespace): command line arguments
    """

    image_paths = get_file_paths(args.input, supported_image_formats)
    xml_paths = [image_path_to_xml_path(image_path) for image_path in image_paths]

    # xml_paths = get_file_paths(args.input, [".xml"])

    # Single thread
    # regions_per_page = []
    # for xml_path_i in tqdm(xml_paths):
    #     regions_per_page.extend(count_regions_single_page(xml_path_i))

    # Multithread
    with Pool(os.cpu_count()) as pool:
        regions_per_page = list(
            tqdm(
                iterable=pool.imap_unordered(count_regions_single_page, xml_paths),
                total=len(xml_paths),
                desc="Extracting Regions",
            )
        )

    # Combine the counters of multiple regions
    total_regions = sum(regions_per_page, Counter())

    pretty_print(dict(total_regions), n_decimals=0)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
