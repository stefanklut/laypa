import cv2
import numpy as np
from detectron2.config import configurable

from page_xml.xml_converters.xml_converter import _XMLConverter
from page_xml.xmlPAGE import PageData


class XMLToBinarySeg(_XMLConverter):
    """
    New binary_seg functions must be of the form:
    def build_{mode}(self, page: PageData, out_size: tuple[int, int]) -> np.ndarray:
    Where mode is the name of the mode in the xml_regions. See XMLConverter for more info
    """

    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

    def build_baseline(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the binary seg version of the baselines
        """
        baseline_color = 1
        size = page.get_size()
        sem_seg = np.zeros((*out_size, 1), np.uint8)
        total_overlap = False
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            sem_seg[..., 0], overlap = self.draw_line(
                sem_seg[..., 0], coords, baseline_color, thickness=self.xml_regions.line_width
            )
            total_overlap = total_overlap or overlap

        if total_overlap:
            self.logger.warning(f"File {page.filepath} contains overlapping baseline sem_seg")
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baseline sem_seg")
        return sem_seg
