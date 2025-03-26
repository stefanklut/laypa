from typing import TypedDict

import cv2
import numpy as np
from detectron2.config import configurable

from page_xml.pageXML_editor import PageXMLEditor
from page_xml.xml_converters.xml_converter import _XMLConverter


class SegmentsInfo(TypedDict):
    """
    Required fields for an segments info dict
    """

    id: int
    category_id: int
    iscrowd: bool


class XMLToPano(_XMLConverter):
    """
    New pano functions must be of the form:
    def build_{mode}(self, page: PageXMLEditor, out_size: tuple[int, int]) -> tuple[np.ndarray, list]:
    Where mode is the name of the mode in the xml_regions. See XMLConverter for more info
    """

    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

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

    def build_baseline(self, page: PageXMLEditor, out_size: tuple[int, int]):
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
            assert isinstance(rgb_color, tuple), "RGB color must be a tuple"
            assert len(rgb_color) == 3, "RGB color must have 3 values"
            pano_mask, overlap = self.draw_line(pano_mask, coords, rgb_color, thickness=self.xml_regions.line_width)
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

    def build_region(self, page: PageXMLEditor, out_size: tuple[int, int]):
        """
        Create the pano version of the regions
        """
        size = page.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for element in set(self.xml_regions.region_types.values()):
            for element_class, element_coords in page.iter_class_coords(element, self.xml_regions.regions_to_classes):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                rgb_color = self.id2rgb(_id)
                assert isinstance(rgb_color, tuple), "RGB color must be a tuple"
                assert len(rgb_color) == 3, "RGB color must have 3 values"
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

    def build_text_line(self, page: PageXMLEditor, out_size: tuple[int, int]):
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
            assert isinstance(rgb_color, tuple), "RGB color must be a tuple"
            assert len(rgb_color) == 3, "RGB color must have 3 values"
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
