from typing import Any, TypedDict

import cv2
import numpy as np
from detectron2 import structures
from detectron2.config import configurable

from page_xml.xml_converters.xml_converter import _XMLConverter
from page_xml.xmlPAGE import PageData


class Annotation(TypedDict):
    """
    Required fields for an annotation dict
    """

    bbox: list[float]
    category_id: int


class XMLToCOCO(_XMLConverter):
    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

    def _bounding_box_center(self, array: np.ndarray) -> list[float]:
        min_x, min_y = np.min(array, axis=0)
        max_x, max_y = np.max(array, axis=0)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        bbox = np.asarray([center_x, center_y, width, height]).astype(np.float32).tolist()
        return bbox

    def build_region(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the instance version of the regions
        """
        out_size = (1, 1)
        size = page.get_size()
        annotations = []
        for element in set(self.xml_regions.region_types.values()):
            for element_class, element_coords in page.iter_class_coords(element, self.xml_regions.region_classes):
                coords = self._scale_coords(element_coords, out_size, size)
                bbox = self._bounding_box_center(coords)
                category_id = element_class - 1  # Ignore background class

                annotation: Annotation = {
                    "bbox": bbox,
                    "category_id": category_id,
                }
                annotations.append(annotation)
        if not annotations:
            self.logger.warning(f"File {page.filepath} does not contains region instances")
        output = {
            "annotations": annotations,
            "image_size": out_size,
        }
        return output
