from typing import TypedDict

import cv2
import numpy as np
from detectron2 import structures
from detectron2.config import configurable

from page_xml.page_xml_editor import PageXMLEditor
from page_xml.xml_converters.xml_converter import _XMLConverter


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


class XMLToInstances(_XMLConverter):
    """
    New instance functions must be of the form:
    def build_{mode}(self, page: PageXMLEditor, out_size: tuple[int, int]) -> np.ndarray:
    Where mode is the name of the mode in the xml_regions. See XMLConverter for more info
    """

    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

    def build_region(self, page: PageXMLEditor, out_size: tuple[int, int]) -> list[Instance]:
        """
        Create the instance version of the regions
        """
        size = page.get_size()
        instances = []
        for element in set(self.xml_regions.region_types.values()):
            for element_class, element_coords in page.iter_class_coords(element, self.xml_regions.regions_to_classes):
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

    def build_instances_text_line(self, page: PageXMLEditor, out_size: tuple[int, int]) -> list[Instance]:
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

    def build_baseline(self, page: PageXMLEditor, out_size: tuple[int, int]) -> list[Instance]:
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
            # HACK Currently the most simple quickest solution used can probably be optimized
            mask, _ = self.draw_line(mask, coords, 255, thickness=self.xml_regions.line_width)
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
