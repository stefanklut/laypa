import cv2
import numpy as np
from detectron2.config import CfgNode, configurable

from page_xml.xml_converters.xml_converter import _XMLConverter
from page_xml.xmlPAGE import PageData
from utils.vector_utils import point_top_bottom_assignment


class XMLToSemSeg(_XMLConverter):
    """
    New sem_seg functions must be of the form:
    def build_{mode}(self, page: PageData, out_size: tuple[int, int]) -> np.ndarray:
    Where mode is the name of the mode in the xml_regions. See XMLConverter for more info
    """

    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

    def build_baseline(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the sem_seg version of the baselines
        """
        baseline_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            sem_seg, overlap = self.draw_line(sem_seg, coords, baseline_color, thickness=self.xml_regions.line_width)
            total_overlap = total_overlap or overlap

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline sem_seg")
        return sem_seg

    def build_region(self, page: PageData, out_size: tuple[int, int]):
        """
        Builds a "image" mask of desired elements
        """
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for element in set(self.xml_regions.region_types.values()):
            for element_class, element_coords in page.iter_class_coords(element, self.xml_regions.region_classes):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(sem_seg, [rounded_coords], element_class)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains region sem_seg")
        return sem_seg

    def build_class_baseline(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the sem_seg version of the class baseline (baseline for each class)
        """
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        total_overlap = False
        for element in set(self.xml_regions.region_types.values()):
            for element_class, baseline_coords in page.iter_class_baseline_coords(element, self.xml_regions.region_classes):
                coords = self._scale_coords(baseline_coords, out_size, size)
                sem_seg, overlap = self.draw_line(sem_seg, coords, element_class, thickness=self.xml_regions.line_width)
                total_overlap = total_overlap or overlap

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping class baseline sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains class baseline sem_seg")
        return sem_seg

    def build_text_line(self, page: PageData, out_size: tuple[int, int]):
        """
        Builds a sem_seg mask of the text line
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

    # TOP BOTTOM

    def build_top_bottom(self, page: PageData, out_size: tuple[int, int]):
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
            binary_mask, overlap = self.draw_line(binary_mask, coords, baseline_color, thickness=self.xml_regions.line_width)
            total_overlap = total_overlap or overlap

            # Add single line to full sem_seg
            line_pixel_coords = np.column_stack(np.where(binary_mask == 1))[:, ::-1]
            rounded_coords = np.round(coords).astype(np.int32)
            top_bottom = point_top_bottom_assignment(rounded_coords, line_pixel_coords)
            colored_top_bottom = np.where(top_bottom, top_color, bottom_color)
            sem_seg[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = colored_top_bottom

            binary_mask.fill(0)

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping top bottom sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains top bottom sem_seg")
        return sem_seg

    def build_start(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the sem_seg version of the start
        """
        start_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[0]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, self.xml_regions.line_width, start_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains start sem_seg")
        return sem_seg

    def build_end(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the sem_seg version of the end
        """
        end_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[-1]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, self.xml_regions.line_width, end_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains end sem_seg")
        return sem_seg

    def build_separator(self, page: PageData, out_size: tuple[int, int]):
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
            cv2.circle(sem_seg, coords_start, self.xml_regions.line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(sem_seg, coords_end, self.xml_regions.line_width, separator_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains separator sem_seg")
        return sem_seg

    def build_baseline_separator(self, page: PageData, out_size: tuple[int, int]):
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
            sem_seg, overlap = self.draw_line(sem_seg, rounded_coords, baseline_color, thickness=self.xml_regions.line_width)
            total_overlap = total_overlap or overlap

            coords_start = rounded_coords[0]
            cv2.circle(sem_seg, coords_start, self.xml_regions.line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(sem_seg, coords_end, self.xml_regions.line_width, separator_color, -1)

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline separator sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline separator sem_seg")
        return sem_seg
