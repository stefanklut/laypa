import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, TypedDict

import cv2
import numpy as np
from detectron2 import structures

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_regions import XMLRegions
from page_xml.xmlPAGE import PageData
from utils.logging_utils import get_logger_name
from utils.vector_utils import (
    point_at_start_or_end_assignment,
    point_top_bottom_assignment,
)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLRegions.get_parser()], description="Code to turn an xml file into an array")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output file", required=True, type=str)

    xml_converter_args = parser.add_argument_group("XML Converter")
    xml_converter_args.add_argument("--square-lines", help="Square the lines", action="store_true")

    args = parser.parse_args()
    return args


class Instance(TypedDict):
    """
    Required fields for an instance dict
    """

    bbox: list[float]
    bbox_mode: int
    category_id: int
    segmentation: list[list[float]]
    keypoints: list[float]
    iscrowd: bool


class SegmentsInfo(TypedDict):
    """
    Required fields for an segments info dict
    """

    id: int
    category_id: int
    iscrowd: bool


# IDEA have fixed ordering of the classes, maybe look at what order is best
class XMLConverter:
    """
    Class for turning a pageXML into ground truth with classes (for segmentation)
    """

    def __init__(
        self,
        xml_regions: XMLRegions,
        square_lines: bool = True,
    ) -> None:
        """
        Class for turning a pageXML into an image with classes

        Args:
            xml_regions (XMLRegions): helps to convert page xml regions to images
        """
        self.logger = logging.getLogger(get_logger_name())
        self.xml_regions = xml_regions
        self.square_lines = square_lines

    @staticmethod
    def _scale_coords(coords: np.ndarray, out_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
        scale_factor = np.asarray(out_size) / np.asarray(size)
        scaled_coords = (coords * scale_factor[::-1]).astype(np.float32)
        return scaled_coords

    @staticmethod
    def _bounding_box(array: np.ndarray) -> list[float]:
        min_x, min_y = np.min(array, axis=0)
        max_x, max_y = np.max(array, axis=0)
        bbox = np.asarray([min_x, min_y, max_x, max_y]).astype(np.float32).tolist()
        return bbox

    # Taken from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
    @staticmethod
    def id2rgb(id_map: int | np.ndarray) -> tuple[int, int, int] | np.ndarray:
        if isinstance(id_map, np.ndarray):
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map % 256
                id_map //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return tuple(color)

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

        rounded_coords = np.round(coords).astype(np.int32)

        # Clear the temp image
        temp_image.fill(0)

        if self.square_lines:
            cv2.polylines(temp_image, [rounded_coords.reshape(-1, 1, 2)], False, 1, thickness)
            line_pixel_coords = np.column_stack(np.where(temp_image == 1))[:, ::-1]
            start_or_end = point_at_start_or_end_assignment(rounded_coords, line_pixel_coords)
            colored_start_or_end = np.where(start_or_end, 0, color)
            temp_image[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = colored_start_or_end
        else:
            cv2.polylines(temp_image, [rounded_coords.reshape(-1, 1, 2)], False, color, thickness)

        overlap = np.logical_and(temp_image, image).any().item()
        image = np.where(temp_image == 0, image, temp_image)

        return image, overlap

    ## REGIONS

    def build_region_instances(self, page: PageData, out_size: tuple[int, int], elements, class_dict) -> list[Instance]:
        size = page.get_size()
        instances = []
        for element in elements:
            for element_class, element_coords in page.iter_class_coords(element, class_dict):
                coords = self._scale_coords(element_coords, out_size, size)
                bbox = self._bounding_box(coords)
                bbox_mode = structures.BoxMode.XYXY_ABS
                flattened_coords = coords.flatten().tolist()
                instance: Instance = {
                    "bbox": bbox,
                    "bbox_mode": bbox_mode,
                    "category_id": element_class - 1,  # -1 for not having background as class
                    "segmentation": [flattened_coords],
                    "keypoints": [],
                    "iscrowd": False,
                }
                instances.append(instance)
        if not instances:
            self.logger.warning(f"File {page.filepath} does not contains region instances")
        return instances

    def build_region_pano(self, page: PageData, out_size: tuple[int, int], elements, class_dict):
        """
        Create the pano version of the regions
        """
        size = page.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for element in elements:
            for element_class, element_coords in page.iter_class_coords(element, class_dict):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                rgb_color = self.id2rgb(_id)
                cv2.fillPoly(pano_mask, [rounded_coords], rgb_color)

                segment: SegmentsInfo = {
                    "id": _id,
                    "category_id": element_class - 1,  # -1 for not having background as class
                    "iscrowd": False,
                }
                segments_info.append(segment)

                _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains region pano")
        return pano_mask, segments_info

    def build_region_sem_seg(self, page: PageData, out_size: tuple[int, int], elements, class_dict):
        """
        Builds a "image" mask of desired elements
        """
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for element in elements:
            for element_class, element_coords in page.iter_class_coords(element, class_dict):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(sem_seg, [rounded_coords], element_class)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains region sem_seg")
        return sem_seg

    ## TEXT LINE

    def build_text_line_instances(self, page: PageData, out_size: tuple[int, int]) -> list[Instance]:
        """
        Create the instance version of the text line
        """
        text_line_class = 0
        size = page.get_size()
        instances = []
        for element_coords in page.iter_text_line_coords():
            coords = self._scale_coords(element_coords, out_size, size)
            bbox = self._bounding_box(coords)
            bbox_mode = structures.BoxMode.XYXY_ABS
            flattened_coords = coords.flatten().tolist()
            instance: Instance = {
                "bbox": bbox,
                "bbox_mode": bbox_mode,
                "category_id": text_line_class,
                "segmentation": [flattened_coords],
                "keypoints": [],
                "iscrowd": False,
            }
            instances.append(instance)

        if not instances:
            self.logger.warning(f"File {page.filepath} does not contains text line instances")
        return instances

    def build_text_line_pano(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the pano version of the text line
        """
        text_line_class = 0
        size = page.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for element_coords in page.iter_text_line_coords():
            coords = self._scale_coords(element_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            rgb_color = self.id2rgb(_id)
            cv2.fillPoly(pano_mask, [rounded_coords], rgb_color)

            segment: SegmentsInfo = {
                "id": _id,
                "category_id": text_line_class,
                "iscrowd": False,
            }
            segments_info.append(segment)

            _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains text line pano")
        return pano_mask, segments_info

    def build_text_line_sem_seg(self, page: PageData, out_size: tuple[int, int]):
        """
        Builds a "image" mask of the text line
        """
        text_line_class = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for element_coords in page.iter_text_line_coords():
            coords = self._scale_coords(element_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.fillPoly(sem_seg, [rounded_coords], text_line_class)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains text line sem_seg")
        return sem_seg

    ## BASELINE

    def build_baseline_instances(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the instance version of the baselines
        """
        baseline_class = 0
        size = page.get_size()
        mask = np.zeros(out_size, np.uint8)
        instances = []
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            mask.fill(0)
            # HACK Currenty the most simple quickest solution used can probably be optimized
            mask, _ = self.draw_line(mask, coords, 255, thickness=line_width)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                raise ValueError(f"{page.filepath} has no contours")

            # Multiple contours should really not happen, but when it does it can still be supported
            all_coords = []
            for contour in contours:
                contour_coords = np.asarray(contour).reshape(-1, 2)
                all_coords.append(contour_coords)
            flattened_coords_list = [coords.flatten().tolist() for coords in all_coords]

            bbox = self._bounding_box(np.concatenate(all_coords, axis=0))
            bbox_mode = structures.BoxMode.XYXY_ABS
            instance: Instance = {
                "bbox": bbox,
                "bbox_mode": bbox_mode,
                "category_id": baseline_class,
                "segmentation": flattened_coords_list,
                "keypoints": [],
                "iscrowd": False,
            }
            instances.append(instance)
        if not instances:
            self.logger.warning(f"File {page.filepath} does not contains baseline instances")
        return instances

    def build_baseline_pano(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the pano version of the baselines
        """
        baseline_class = 0
        size = page.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rgb_color = self.id2rgb(_id)
            pano_mask, overlap = self.draw_line(pano_mask, coords, rgb_color, thickness=line_width)
            total_overlap = total_overlap or overlap
            segment: SegmentsInfo = {
                "id": _id,
                "category_id": baseline_class,
                "iscrowd": False,
            }
            segments_info.append(segment)
            _id += 1

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline pano")
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline pano")
        return pano_mask, segments_info

    def build_baseline_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the baselines
        """
        baseline_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            sem_seg, overlap = self.draw_line(sem_seg, coords, baseline_color, thickness=line_width)
            total_overlap = total_overlap or overlap

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline sem_seg")
        return sem_seg

    # TOP BOTTOM

    def build_top_bottom_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the top bottom
        """
        baseline_color = 1
        top_color = 1
        bottom_color = 2
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        binary_mask = np.zeros(out_size, dtype=np.uint8)
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            binary_mask, overlap = self.draw_line(binary_mask, coords, baseline_color, thickness=line_width)
            total_overlap = total_overlap or overlap

            # Add single line to full sem_seg
            line_pixel_coords = np.column_stack(np.where(binary_mask == 1))[:, ::-1]
            rounded_coords = np.round(coords).astype(np.int32)
            top_bottom = point_top_bottom_assignment(rounded_coords, line_pixel_coords)
            colored_top_bottom = np.where(top_bottom, top_color, bottom_color)
            sem_seg[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = colored_top_bottom

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping top bottom sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains top bottom sem_seg")
        return sem_seg

    ## START

    def build_start_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the start
        """
        start_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[0]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, line_width, start_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains start sem_seg")
        return sem_seg

    ## END

    def build_end_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the end
        """
        end_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[-1]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, line_width, end_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains end sem_seg")
        return sem_seg

    ## SEPARATOR

    def build_separator_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the separator
        """
        separator_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            coords_start = rounded_coords[0]
            cv2.circle(sem_seg, coords_start, line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(sem_seg, coords_end, line_width, separator_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains separator sem_seg")
        return sem_seg

    ## BASELINE + SEPARATOR

    def build_baseline_separator_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Create the sem_seg version of the baseline separator
        """
        baseline_color = 1
        separator_color = 2

        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            sem_seg, overlap = self.draw_line(sem_seg, rounded_coords, baseline_color, thickness=line_width)
            total_overlap = total_overlap or overlap

            coords_start = rounded_coords[0]
            cv2.circle(sem_seg, coords_start, line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(sem_seg, coords_end, line_width, separator_color, -1)

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline separator sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline separator sem_seg")
        return sem_seg

    def to_sem_seg(
        self,
        xml_path: Path,
        original_image_shape: Optional[tuple[int, int]] = None,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Turn a single pageXML into a mask of labels

        Args:
            xml_path (Path): path to pageXML
            original_image_shape (tuple, optional): shape of the original image. Defaults to None.
            image_shape (tuple, optional): shape of the output image. Defaults to None.

        Raises:
            NotImplementedError: mode is not known

        Returns:
            Optional[np.ndarray]: mask of labels
        """
        gt_data = PageData(xml_path)
        gt_data.parse()

        if original_image_shape is not None:
            gt_data.set_size(original_image_shape)

        if image_shape is None:
            image_shape = gt_data.get_size()

        if self.xml_regions.mode == "region":
            sem_seg = self.build_region_sem_seg(
                gt_data,
                image_shape,
                set(self.xml_regions.region_types.values()),
                self.xml_regions.region_classes,
            )
            return sem_seg
        elif self.xml_regions.mode == "baseline":
            sem_seg = self.build_baseline_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "top_bottom":
            sem_seg = self.build_top_bottom_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "start":
            sem_seg = self.build_start_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "end":
            sem_seg = self.build_end_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "separator":
            sem_seg = self.build_separator_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "baseline_separator":
            sem_seg = self.build_baseline_separator_sem_seg(
                gt_data,
                image_shape,
                line_width=self.xml_regions.line_width,
            )
            return sem_seg
        elif self.xml_regions.mode == "text_line":
            sem_seg = self.build_text_line_sem_seg(
                gt_data,
                image_shape,
            )
            return sem_seg
        else:
            return None

    def to_instances(
        self,
        xml_path: Path,
        original_image_shape: Optional[tuple[int, int]] = None,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[list]:
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

        if self.xml_regions.mode == "region":
            instances = self.build_region_instances(
                gt_data,
                image_shape,
                set(self.xml_regions.region_types.values()),
                self.xml_regions.region_classes,
            )
            return instances
        elif self.xml_regions.mode == "baseline":
            instances = self.build_baseline_instances(
                gt_data,
                image_shape,
                self.xml_regions.line_width,
            )
            return instances
        elif self.xml_regions.mode == "text_line":
            instances = self.build_text_line_instances(
                gt_data,
                image_shape,
            )
            return instances
        else:
            return None

    def to_pano(
        self,
        xml_path: Path,
        original_image_shape: Optional[tuple[int, int]] = None,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[tuple[np.ndarray, list]]:
        """
        Turn a single pageXML into a pano image with corresponding pixel info

        Args:
            xml_path (Path): path to pageXML
            original_image_shape (Optional[tuple[int, int]], optional): shape of the original image. Defaults to None.
            image_shape (Optional[tuple[int, int]], optional): shape of the output image. Defaults to None.

        Raises:
            NotImplementedError: mode is not known

        Returns:
            Optional[tuple[np.ndarray, list]]: pano mask and the segments information
        """
        gt_data = PageData(xml_path)
        gt_data.parse()

        if original_image_shape is not None:
            gt_data.set_size(original_image_shape)

        if image_shape is None:
            image_shape = gt_data.get_size()

        if self.xml_regions.mode == "region":
            pano, segments_info = self.build_region_pano(
                gt_data,
                image_shape,
                set(self.xml_regions.region_types.values()),
                self.xml_regions.region_classes,
            )
            return pano, segments_info
        elif self.xml_regions.mode == "baseline":
            pano, segments_info = self.build_baseline_pano(
                gt_data,
                image_shape,
                self.xml_regions.line_width,
            )
            return pano, segments_info
        elif self.xml_regions.mode == "text_line":
            pano, segments_info = self.build_text_line_pano(
                gt_data,
                image_shape,
            )
            return pano, segments_info
        else:
            return None


if __name__ == "__main__":
    args = get_arguments()
    xml_regions = XMLRegions(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type,
    )
    XMLConverter(xml_regions, args.square_lines)

    input_path = Path(args.input)
    output_path = Path(args.output)
