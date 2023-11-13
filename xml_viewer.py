import argparse
import logging
import os
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer

# from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm

from core.setup import setup_cfg
from datasets.dataset import metadata_from_classes
from page_xml.xml_converter import XMLConverter
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import get_file_paths
from utils.logging_utils import get_logger_name
from utils.path_utils import xml_path_to_image_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize XML files for visualization/debugging")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/files", nargs="+", default=[], required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)

    parser.add_argument(
        "-t", "--output_type", help="Output mode", choices=["gray", "color", "overlay"], default="overlay", type=str
    )

    args = parser.parse_args()
    return args


class Viewer:
    # TODO Resize the output image to a specified size / percentage, maybe use ResizeShortestEdge
    """
    Simple viewer to convert xml files to images (grayscale, color or overlay)
    """

    def __init__(
        self,
        xml_converter: XMLConverter,
        output_dir: str | Path,
        output_type="gray",
    ) -> None:
        """
        Simple viewer to convert xml files to images (grayscale, color or overlay)

        Args:
            xml_to_image (XMLImage): converter from xml to a label mask
            output_dir (str | Path): path to output dir
            mode (str, optional): output mode: gray, color, or overlay. Defaults to 'gray'.

        Raises:
            ValueError: Colors do not match the number of region types
            NotImplementedError: Different mode specified than allowed
        """

        self.logger = logging.getLogger(get_logger_name())

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        self.output_dir: Path = output_dir
        self.xml_converter: XMLConverter = xml_converter

        region_names = xml_converter.get_regions()
        self.metadata = metadata_from_classes(region_names)

        self.output_type = output_type

        if self.output_type == "gray":
            self.save_function = self.save_gray_image
        elif self.output_type == "color":
            self.save_function = self.save_color_image
        elif self.output_type == "overlay":
            self.save_function = self.save_overlay_image
        else:
            raise NotImplementedError

    def save_gray_image(self, xml_path_i: Path):
        """
        Save the pageXML as a grayscale image

        Args:
            xml_path_i (Path): single pageXML path
        """
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".png")
        gray_image = self.xml_converter.to_sem_seg(xml_path_i)

        if gray_image is None:
            raise ValueError

        save_image_array_to_path(str(output_image_path), gray_image)

    def save_color_image(self, xml_path_i: Path):
        """
        Save the pageXML as a color image

        Args:
            xml_path_i (Path): single pageXML path
        """
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".png")
        gray_image = self.xml_converter.to_sem_seg(xml_path_i)

        if gray_image is None:
            raise ValueError

        color_image = np.empty((*gray_image.shape, 3), dtype=np.uint8)

        colors = self.metadata.get("stuff_colors")
        assert colors is not None, "Can't make color images without colors"
        assert np.max(gray_image) < len(colors), "Not enough colors, grayscale has too many classes"

        for i, color in enumerate(colors):
            color_image[gray_image == i] = np.asarray(color).reshape((1, 1, 3))

        save_image_array_to_path(str(output_image_path), color_image)

    def save_overlay_image(self, xml_path_i: Path):
        """
        Save the pageXML as a overlay image. Requires the image file to exist folder up

        Args:
            xml_path_i (Path): single pageXML path
        """
        output_image_path = self.output_dir.joinpath(xml_path_i.stem + ".jpg")
        gray_image = self.xml_converter.to_sem_seg(xml_path_i)

        image_path_i = xml_path_to_image_path(xml_path_i)

        image = load_image_array_from_path(str(image_path_i))
        if image is None:
            return

        vis_im = Visualizer(image.copy(), metadata=self.metadata, scale=1)
        vis_im = vis_im.draw_sem_seg(gray_image, alpha=0.4)
        overlay_image = vis_im.get_image()
        save_image_array_to_path(str(output_image_path), overlay_image)

    @staticmethod
    def check_image_exists(xml_paths: list[Path]):
        all(xml_path_to_image_path(xml_path) for xml_path in xml_paths)

    def run(self, xml_list: list[str] | list[Path]) -> None:
        """
        Run the conversion of pageXML to images on all pageXML paths specified

        Args:
            xml_list (list[str] | list[Path]): multiple pageXML paths
        """
        cleaned_xml_list: list[Path] = [Path(path) if isinstance(path, str) else path for path in xml_list]
        cleaned_xml_list = [path.resolve() for path in cleaned_xml_list]

        # with overlay all images must exist as well
        if self.output_type == "overlay":
            self.check_image_exists(cleaned_xml_list)

        if not self.output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({self.output_dir}), creating one at specified location")
            self.output_dir.mkdir(parents=True)

        # Single thread
        # for xml_path_i in tqdm(cleaned_xml_list):
        #     self.save_function(xml_path_i)

        # Multithread
        with Pool(os.cpu_count()) as pool:
            _ = list(
                tqdm(
                    iterable=pool.imap_unordered(self.save_function, cleaned_xml_list),
                    total=len(cleaned_xml_list),
                    desc="Creating Images",
                )
            )


def main(args) -> None:
    cfg = setup_cfg(args)

    xml_list = get_file_paths(args.input, formats=[".xml"])

    xml_converter = XMLConverter(
        mode=cfg.MODEL.MODE,
        line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
        regions=cfg.PREPROCESS.REGION.REGIONS,
        merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
        region_type=cfg.PREPROCESS.REGION.REGION_TYPE,
    )

    viewer = Viewer(xml_converter=xml_converter, output_dir=args.output, output_type=args.output_type)
    viewer.run(xml_list)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
