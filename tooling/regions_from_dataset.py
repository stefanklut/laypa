import argparse

# from multiprocessing.pool import ThreadPool as Pool
import os
import sys
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from xml_comparison import pretty_print

from page_xml.page_xml_editor import PageXMLEditor
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.path_utils import image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count regions from a dataset")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)

    output_args = parser.add_argument_group("Output")
    output_args.add_argument("--show_filenames", action="store_true", help="Show filenames of pages with issues")
    args = parser.parse_args()
    return args


def count_regions_single_page(xml_path: Path) -> dict[str, Any]:
    """
    Count the unique regions in a PageXML

    Args:
        xml_path (Path): Path to PageXML

    Returns:
        Counter: Count of all unique regions
    """
    page_data = PageXMLEditor(xml_path)

    region_names = ["TextRegion"]  # Assuming this is all there is
    zones = page_data.get_zones(region_names)

    if zones is None:
        return {"path": xml_path, "count": Counter()}

    counter = Counter(item["type"] for item in zones.values())
    return {"path": xml_path, "count": counter}


def main(args):
    """
    Run the full count over all PageXMLs found in the input dir

    Args:
        args (argparse.Namespace): command line arguments
    """

    image_paths = get_file_paths(args.input, SUPPORTED_IMAGE_FORMATS)
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

    no_regions_pages = []
    none_regions_pages = []
    text_regions_pages = []

    # Combine the counters of multiple regions
    total_regions = Counter()
    for regions in regions_per_page:
        total_regions.update(regions["count"])

        # Check if the page has no regions
        if len(regions["count"]) == 0:
            no_regions_pages.append(regions["path"])
        if None in regions["count"]:
            none_regions_pages.append(regions["path"])
        if "Text" in regions["count"]:
            text_regions_pages.append(regions["path"])

    def print_xml_path_with_image_path(xml_paths: list[Path]) -> None:
        from utils.path_utils import xml_path_to_image_path

        for xml_path in xml_paths:
            image_path = xml_path_to_image_path(xml_path)
            print(f"XML: {xml_path}, Image: {image_path}")

    # count pages that have no region (This could indicate that annotation was not done)
    print(f"Number of pages: {len(regions_per_page)}")
    print(f"Number of pages without regions: {len(no_regions_pages)}")
    if args.show_filenames:
        print_xml_path_with_image_path(no_regions_pages)

    # count pages that have None as a region (None means the region had no annotation)
    print(f"Number of pages with None regions: {len(none_regions_pages)}")
    if args.show_filenames:
        print_xml_path_with_image_path(none_regions_pages)

    # count pages that have Text as a region (Text is the default region so possibly not given an annotation)
    print(f"Number of pages with Text regions: {len(text_regions_pages)}")
    if args.show_filenames:
        print_xml_path_with_image_path(text_regions_pages)

    print(
        f"Number of pages with no issue: {len(regions_per_page) - len(no_regions_pages) - len(none_regions_pages) - len(text_regions_pages)}"
    )
    pretty_print(dict(total_regions), n_decimals=0)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
