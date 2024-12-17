import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from detectron2.config import CfgNode, configurable

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_regions import XMLRegions
from page_xml.xmlPAGE import PageData
from utils.image_utils import save_image_array_to_path
from utils.logging_utils import get_logger_name
from utils.vector_utils import point_at_start_or_end_assignment


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLRegions.get_parser()], description="Code to turn an xml file into an array")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output file", required=True, type=str)

    xml_converter_args = parser.add_argument_group("XML Converter")
    xml_converter_args.add_argument("--square-lines", help="Square the lines", action="store_true")

    args = parser.parse_args()
    return args


# IDEA have fixed ordering of the classes, maybe look at what order is best
class _XMLConverter:
    """
    Base class for converting xml files to other formats, add new converters by subclassing this class and adding a build_{mode} function
    """

    @configurable
    def __init__(
        self,
        xml_regions: XMLRegions,
        square_lines: bool = True,
    ) -> None:
        """
        Initializes an instance of the XMLConverter class.

        Args:
            xml_regions (XMLRegions): An instance of the XMLRegions class that helps to convert page xml regions to images.
            square_lines (bool, optional): A boolean value indicating whether to square the lines in the image. Defaults to True.
        """
        self.logger = logging.getLogger(get_logger_name())
        self.xml_regions = xml_regions
        self.square_lines = square_lines

    @classmethod
    def from_config(cls, cfg: CfgNode) -> dict[str, Any]:
        """
        Converts a configuration object to a dictionary to be used as keyword arguments.

        Args:
            cfg (CfgNode): The configuration object.

        Returns:
            dict[str, Any]: A dictionary containing the converted configuration values.
        """
        ret = {
            "xml_regions": XMLRegions(cfg),
            "square_lines": cfg.PREPROCESS.BASELINE.SQUARE_LINES,
        }
        return ret

    def convert(
        self,
        xml_path: Path,
        original_image_shape: Optional[tuple[int, int]] = None,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Any:
        """
        Turn a single pageXML into a dict with scaled coordinates

        Args:
            xml_path (Path): path to pageXML
            original_image_shape (Optional[tuple[int, int]], optional): shape of the original image. Defaults to None.
            image_shape (Optional[tuple[int, int]], optional): shape of the output image. Defaults to None.

        Raises:
            NotImplementedError: mode is not known

        Returns:
            Optional[dict]: scaled coordinates about the location of the objects in the image
        """
        gt_data = PageData(xml_path)
        gt_data.parse()

        if original_image_shape is not None:
            gt_data.set_size(original_image_shape)

        if image_shape is None:
            image_shape = gt_data.get_size()

        if hasattr(self, f"build_{self.xml_regions.mode}"):
            output = getattr(self, f"build_{self.xml_regions.mode}")(
                gt_data,
                image_shape,
            )
            return output
        else:
            return None

    @staticmethod
    def _scale_coords(coords: np.ndarray, out_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
        scale_factor = (np.asarray(out_size) - 1) / (np.asarray(size) - 1)
        scaled_coords = (coords * scale_factor[::-1]).astype(np.float32)
        return scaled_coords

    @staticmethod
    def _bounding_box(array: np.ndarray) -> list[float]:
        min_x, min_y = np.min(array, axis=0)
        max_x, max_y = np.max(array, axis=0)
        bbox = np.asarray([min_x, min_y, max_x, max_y]).astype(np.float32).tolist()
        return bbox

    def draw_line(
        self,
        image: np.ndarray,
        coords: np.ndarray,
        color: int | tuple[int, int, int],
        thickness: int = 1,
    ) -> tuple[np.ndarray, bool]:
        """
        Draw lines on an image

        Args:
            image (np.ndarray): image to draw on
            lines (np.ndarray): lines to draw
            color (tuple[int, int, int]): color of the lines
            thickness (int, optional): thickness of the lines. Defaults to 1.
        """
        temp_image = np.zeros_like(image)
        if isinstance(color, tuple) and len(color) == 3 and temp_image.ndim == 2:
            raise ValueError("Color should be a single int")

        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        rounded_coords = np.round(coords).astype(np.int32)

        if self.square_lines:
            cv2.polylines(binary_mask, [rounded_coords.reshape(-1, 1, 2)], False, 1, thickness)
            line_pixel_coords = np.column_stack(np.where(binary_mask == 1))[:, ::-1]
            start_or_end = point_at_start_or_end_assignment(rounded_coords, line_pixel_coords)
            if temp_image.ndim == 3:
                start_or_end = np.expand_dims(start_or_end, axis=1)
            colored_start_or_end = np.where(start_or_end, 0, color)
            temp_image[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = colored_start_or_end
        else:
            cv2.polylines(temp_image, [rounded_coords.reshape(-1, 1, 2)], False, color, thickness)

        overlap = np.logical_and(temp_image, image).any().item()
        image = np.where(temp_image == 0, image, temp_image)

        return image, overlap


if __name__ == "__main__":
    from page_xml.xml_converters.xml_to_sem_seg import XMLToSemSeg

    args = get_arguments()
    xml_regions = XMLRegions(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type,
    )
    xml_converter = XMLToSemSeg(xml_regions, args.square_lines)

    input_path = Path(args.input)
    output_path = Path(args.output)

    sem_seg = xml_converter.convert(
        input_path,
        original_image_shape=None,
        image_shape=None,
    )

    # save image
    save_image_array_to_path(output_path, sem_seg)
