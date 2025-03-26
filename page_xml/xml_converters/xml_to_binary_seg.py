import cv2
import numpy as np
from detectron2.config import configurable

from page_xml.pageXML_editor import PageXMLEditor
from page_xml.xml_converters.xml_converter import _XMLConverter


class XMLToBinarySeg(_XMLConverter):
    """
    New binary_seg functions must be of the form:
    def build_{mode}(self, page: PageXMLEditor, out_size: tuple[int, int]) -> np.ndarray:
    Where mode is the name of the mode in the xml_regions. See XMLConverter for more info
    """

    @configurable
    def __init__(self, xml_regions, square_lines):
        super().__init__(xml_regions, square_lines)

    def build_baseline(self, page: PageXMLEditor, out_size: tuple[int, int]):
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

    def build_region(self, page: PageXMLEditor, out_size: tuple[int, int]):
        """
        Builds a "image" mask of desired elements
        """
        size = page.get_size()
        n_classes = len(self.xml_regions.regions) - 1  # -1 because we don't count the background
        sem_seg = []
        for _ in range(n_classes):
            sem_seg.append(np.zeros((*out_size, 1), np.uint8))

        for element in set(self.xml_regions.region_types.values()):
            for element_class, element_coords in page.iter_class_coords(element, self.xml_regions.regions_to_classes):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(sem_seg[element_class - 1], [rounded_coords], (1,))

        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, n_classes)
        # for i, ax in enumerate(axes):
        #     ax.imshow(sem_seg[i].squeeze())
        #     ax.set_title(self.xml_regions.regions[i + 1])
        # plt.show()

        sem_seg = np.concatenate(sem_seg, axis=-1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains region sem_seg")
        return sem_seg
