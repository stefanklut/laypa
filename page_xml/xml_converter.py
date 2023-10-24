import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import cv2
from detectron2 import structures
from typing import Optional, TypedDict

from utils.vector_utils import point_top_bottom_assignment

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xmlPAGE import PageData
from page_xml.xml_regions import XMLRegions
from utils.logging_utils import get_logger_name


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLConverter.get_parser()],
        description="Code to turn an xml file into an array")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file",
                        required=True, type=str)
    io_args.add_argument(
        "-o", "--output", help="Output file", required=True, type=str)

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
class XMLConverter(XMLRegions):
    """
    Class for turning a pageXML into ground truth with classes (for segmentation)
    """
    def __init__(self, mode: str, 
                 line_width: Optional[int] = None,
                 regions: Optional[list[str]] = None, 
                 merge_regions: Optional[list[str]] = None, 
                 region_type: Optional[list[str]] = None) -> None:
        """
        Class for turning a pageXML into an image with classes
        
        Args:
            mode (str): mode of the region type
            line_width (Optional[int], optional): width of line. Defaults to None.
            regions (Optional[list[str]], optional): list of regions to extract from pageXML. Defaults to None.
            merge_regions (Optional[list[str]], optional): list of region to merge into one. Defaults to None.
            region_type (Optional[list[str]], optional): type of region for each region. Defaults to None.
        """
        super().__init__(mode, line_width, regions, merge_regions, region_type)
        self.logger = logging.getLogger(get_logger_name())
        
    @staticmethod            
    def _scale_coords(coords: np.ndarray, out_size: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
        scale_factor = np.asarray(out_size) / np.asarray(size)
        scaled_coords = (coords * scale_factor[::-1]).astype(np.float32)
        return scaled_coords
    
    @staticmethod
    def _bounding_box(array: np.ndarray) -> list[float]:
        min_x, min_y = np.min(array, axis=0)
        max_x, max_y = np.max(array, axis=0)
        bbox =  np.asarray([min_x, min_y, max_x, max_y]).astype(np.float32).tolist()
        return bbox
    
    # Taken from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
    @staticmethod
    def id2rgb(id_map: int|np.ndarray) -> tuple|np.ndarray:
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
                    "bbox"        : bbox,
                    "bbox_mode"   : bbox_mode,
                    "category_id" : element_class - 1, # -1 for not having background as class
                    "segmentation": [flattened_coords],
                    "keypoints"   : [],
                    "iscrowd"     : False
                }
                instances.append(instance)
        return instances
    
    def build_region_pano(self, page: PageData, out_size: tuple[int,int], elements, class_dict):
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
                    "id"         : _id,
                    "category_id": element_class - 1, # -1 for not having background as class
                    "iscrowd"    : False
                }
                segments_info.append(segment)
                
                _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains regions")
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
            self.logger.warning(f"File {page.filepath} does not contains regions")
        return sem_seg
    
    ## TEXT LINE
    
    def build_text_line_instances(self, page: PageData, out_size: tuple[int, int]) -> list[Instance]:
        text_line_class = 0
        size = page.get_size()
        instances = []
        for element_coords in page.iter_text_line_coords():
            coords = self._scale_coords(element_coords, out_size, size)
            bbox = self._bounding_box(coords)
            bbox_mode = structures.BoxMode.XYXY_ABS
            flattened_coords = coords.flatten().tolist()
            instance: Instance = {
                "bbox"        : bbox,
                "bbox_mode"   : bbox_mode,
                "category_id" : text_line_class,
                "segmentation": [flattened_coords],
                "keypoints"   : [],
                "iscrowd"     : False
            }
            instances.append(instance)
        return instances
    
    def build_text_line_pano(self, page: PageData, out_size: tuple[int, int]):
        """
        Create the pano version of the textline
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
                "id"         : _id,
                "category_id": text_line_class,
                "iscrowd"    : False
            }
            segments_info.append(segment)
            
            _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains regions")
        return pano_mask, segments_info
        
    def build_text_line_sem_seg(self, page: PageData, out_size: tuple[int, int]):
        """
        Builds a "image" mask of desired elements
        """
        text_line_class = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for element_coords in page.iter_text_line_coords():
            coords = self._scale_coords(element_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.fillPoly(sem_seg, [rounded_coords], text_line_class)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains regions")
        return sem_seg
    
    ## BASELINE
    
    def build_baseline_instances(self, page: PageData, out_size: tuple[int, int], line_width: int):
        baseline_class = 0
        size = page.get_size()
        mask = np.zeros(out_size, np.uint8)
        instances = []
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            mask.fill(0)
            # HACK Currenty the most simple quickest solution used can probably be optimized
            cv2.polylines(mask, [rounded_coords.reshape(-1, 1, 2)], False, 255, line_width)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                raise ValueError(f"{page.filepath} has no contours")

            # Multiple contours should really not happen, but when it does it can still be supported
            all_coords = []
            for contour in contours:
                contour_coords = np.asarray(contour).reshape(-1,2)
                all_coords.append(contour_coords)
            flattened_coords_list = [coords.flatten().tolist() for coords in all_coords]
            
            bbox = self._bounding_box(np.concatenate(all_coords, axis=0))
            bbox_mode = structures.BoxMode.XYXY_ABS
            instance: Instance = {
                "bbox"        : bbox,
                "bbox_mode"   : bbox_mode,
                "category_id" : baseline_class,
                "segmentation": flattened_coords_list,
                "keypoints"   : [],
                "iscrowd"     : False
            }
            instances.append(instance)
            
        return instances
        
    def build_baseline_pano(self, page: PageData, out_size: tuple[int, int], line_width: int):
        baseline_class = 0
        size = page.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            rgb_color = self.id2rgb(_id)
            cv2.polylines(pano_mask, [rounded_coords.reshape(-1, 1, 2)], False, rgb_color, line_width)
            segment: SegmentsInfo = {
                "id"         : _id,
                "category_id": baseline_class,
                "iscrowd"    : False
            }
            segments_info.append(segment)
            _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return pano_mask, segments_info
    
    def build_baseline_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Baselines on XML-PAGE
        """
        baseline_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.polylines(sem_seg, [rounded_coords.reshape(-1, 1, 2)], False, baseline_color, line_width)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return sem_seg
    
    #TOP BOTTOM
    
    def build_top_bottom_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Baselines Top Bottom on XML-PAGE
        """
        baseline_color = 1
        top_color = 1
        bottom_color = 2
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        binary_mask = np.zeros(out_size, dtype=np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            binary_mask.fill(0)
            cv2.polylines(binary_mask, [rounded_coords.reshape(-1, 1, 2)], False, baseline_color, line_width)
            
            # Add single line to full sem_seg
            line_pixel_coords = np.column_stack(np.where(binary_mask == 1))[:, ::-1]
            top_bottom = point_top_bottom_assignment(rounded_coords, line_pixel_coords)
            colored_top_bottom = np.where(top_bottom, top_color, bottom_color)
            sem_seg[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = colored_top_bottom
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains top bottom sem_seg")
        return sem_seg
    
    
    ## START
    
    def build_start_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Starts on XML-PAGE
        """
        start_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[0]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, line_width, start_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return sem_seg
    
    ## END
    
    def build_end_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Ends on XML-PAGE
        """
        end_color = 1
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[-1]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(sem_seg, rounded_coords, line_width, end_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return sem_seg
    
    ## SEPARATOR
    
    def build_separator_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Separators on XML-PAGE
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
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return sem_seg
    
    ## BASELINE + SEPARATOR
    
    def build_baseline_separator_sem_seg(self, page: PageData, out_size: tuple[int, int], line_width: int):
        """
        Builds a "image" mask of Separators and Baselines on XML-PAGE
        """
        baseline_color = 1
        separator_color = 2
        
        size = page.get_size()
        sem_seg = np.zeros(out_size, np.uint8)
        for baseline_coords in page.iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.polylines(sem_seg, [coords.reshape(-1, 1, 2)], False, baseline_color, line_width)
            
            coords_start = rounded_coords[0]
            cv2.circle(sem_seg, coords_start, line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(sem_seg, coords_end, line_width, separator_color, -1)
        if not sem_seg.any():
            self.logger.warning(f"File {page.filepath} does not contains baselines")
        return sem_seg

    def to_sem_seg(self, 
                   xml_path: Path, 
                   original_image_shape: Optional[tuple[int, int]]=None, 
                   image_shape: Optional[tuple[int, int]]=None) -> Optional[np.ndarray]:
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
            
        if self.mode == "region":
            sem_seg = self.build_region_sem_seg(
                gt_data,
                image_shape,
                set(self.region_types.values()),
                self.region_classes
            )
            return sem_seg
        elif self.mode == "baseline":
            sem_seg = self.build_baseline_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "top_bottom":
            sem_seg = self.build_top_bottom_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "start":
            sem_seg = self.build_start_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "end":
            sem_seg = self.build_end_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "separator":
            sem_seg = self.build_separator_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "baseline_separator":
            sem_seg = self.build_baseline_separator_sem_seg(
                gt_data,
                image_shape,
                line_width=self.line_width
            )
            return sem_seg
        elif self.mode == "text_line":
            sem_seg = self.build_text_line_sem_seg(
                gt_data,
                image_shape
            )
            return sem_seg
        else:
            return None

    def to_instances(self, 
                     xml_path: Path, 
                     original_image_shape: Optional[tuple[int, int]]=None, 
                     image_shape: Optional[tuple[int, int]]=None) -> Optional[list]:
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
            
        if self.mode == "region":
            instances = self.build_region_instances(
                gt_data,
                image_shape,
                set(self.region_types.values()),
                self.region_classes
            )
            return instances
        elif self.mode == "baseline":
            instances = self.build_baseline_instances(
                gt_data,
                image_shape,
                self.line_width
            )
            return instances
        elif self.mode == "text_line":
            instances = self.build_text_line_instances(
                gt_data,
                image_shape
            )
            return instances
        else:
            return None
        
    def to_pano(self, 
                xml_path: Path, 
                original_image_shape: Optional[tuple[int, int]]=None, 
                image_shape: Optional[tuple[int, int]]=None) -> Optional[tuple[np.ndarray, list]]:
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
            
        if self.mode == "region":
            pano, segments_info = self.build_region_pano(
                gt_data,
                image_shape,
                set(self.region_types.values()),
                self.region_classes
            )
            return pano, segments_info
        elif self.mode == "baseline":
            pano, segments_info = self.build_baseline_pano(
                gt_data,
                image_shape,
                self.line_width
            )
            return pano, segments_info
        elif self.mode == "text_line":
            pano, segments_info = self.build_text_line_pano(
                gt_data,
                image_shape
            )
            return pano, segments_info
        else:
            return None

if __name__ == "__main__":
    args = get_arguments()
    XMLConverter(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    
    
    
