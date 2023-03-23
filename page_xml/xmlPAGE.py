# Modified from P2PaLA

import os
import logging
from typing import TypedDict

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import re
import datetime
from pathlib import Path
from detectron2 import structures

class Instance(TypedDict):
    bbox: list[float]
    bbox_mode: int
    category_id: int
    segmentation: list[list[float]]
    keypoints: list[float]
    iscrowd: bool
    
class SegmentsInfo(TypedDict):
    id: int
    category_id: int
    iscrowd: bool

class PageData:
    """ Class to process PAGE xml files"""

    def __init__(self, filepath: Path, logger=None, creator=None):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.logger = logging.getLogger(__name__) if logger == None else logger
        self.filepath = filepath
        self.name = self.filepath.stem
        self.creator = "Laypa" if creator == None else creator
        
        # REVIEW should this be replaced with the newer pageXML standard?
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    " http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }
        self.size = None
        # self.parse()
    
    def set_size(self, size: tuple[int, int]):
        self.size = size
    
    def parse(self):
        """
        Parse PAGE-XML file
        """
        tree = ET.parse(self.filepath)
        # --- get the root of the data
        self.root = tree.getroot()
        # --- save "namespace" base
        self.base = "".join([self.root.tag.rsplit("}", 1)[0], "}"])

    def get_region(self, region_name):
        """
        get all regions in PAGE which match region_name
        """
        return self.root.findall("".join([".//", self.base, region_name])) or None

    def get_zones(self, region_names):
        to_return = {}
        idx = 0
        for element in region_names:
            for node in self.root.findall("".join([".//", self.base, element])):
                to_return[idx] = {'coords':self.get_coords(node),
                        'type': self.get_region_type(node),
                        'id':self.get_id(node)} 
                idx += 1
        if to_return:
            return to_return
        else:
            return None


    def get_id(self, element):
        """
        get Id of current element
        """
        return str(element.attrib.get("id"))

    def get_region_type(self, element): 
        """
        Returns the type of element
        """
        re_match = re.match(r".*structure {.*type:(.*);.*}", element.attrib["custom"])
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        e_type = re_match.group(1)
        
        return e_type

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        if self.size is not None:
            return self.size
        
        img_width = int(
            self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageWidth")
        )
        img_height = int(
            self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageHeight")
        )
        self.size = (img_height, img_width)
        
        return self.size

    def get_coords(self, element):
        str_coords = (
            element.findall("".join(["./", self.base, "Coords"]))[0]
            .attrib.get("points")
            .split()
        )
        return np.array([i.split(",") for i in str_coords]).astype(np.int32)

    def get_polygons(self, element_name):
        """
        returns a list of polygons for the element desired
        """
        polygons = []
        for element in self._iter_element(element_name):
            # --- get element type
            e_type = self.get_region_type(element)
            if e_type == None:
                self.logger.warning(
                    f"Element type \"{e_type}\" undefined, set to \"other\""
                )
                e_type = "other"

            polygons.append([self.get_coords(element), e_type])

        return polygons
    
    def _iter_element(self, element):
        return self.root.iterfind("".join([".//", self.base, element]))
    
    def _iter_class_coords(self, element, class_dict):
        for node in self._iter_element(element):
            element_type = self.get_region_type(node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(
                    f"Element type \"{element_type}\" undefined in class dict {self.filepath}"
                )
                continue
            element_class = class_dict[element_type]
            element_coords = self.get_coords(node)
            
            # Ignore lines
            if element_coords.shape[0] < 3:
                continue
            
            yield element_class, element_coords
            
    def _iter_baseline_coords(self):
        for node in self._iter_element("Baseline"):
            str_coords = node.attrib.get("points")
            if str_coords is None:
                continue
            split_str_coords = str_coords.split()
            # REVIEW currently ignoring empty baselines
            if len(split_str_coords) == 0:
                continue
            coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
            yield coords
            
    def _iter_text_line_coords(self):
        for node in self._iter_element("TextLine"):
            coords = self.get_coords(node)
            yield coords

    @staticmethod            
    def _scale_coords(coords: np.ndarray, out_size: tuple[int, int], size: tuple[int, int]):
        scale_factor = np.asarray(out_size) / np.asarray(size)
        scaled_coords = (coords * scale_factor[::-1]).astype(np.float32)
        return scaled_coords
    
    @staticmethod
    def _bounding_box(array: np.ndarray):
        min_x, min_y = np.min(array, axis=0)
        max_x, max_y = np.max(array, axis=0)
        bbox =  np.asarray([min_x, min_y, max_x, max_y]).astype(np.float32).tolist()
        return bbox
    
    # Taken from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
    @staticmethod
    def id2rgb(id_map):
        if isinstance(id_map, np.ndarray):
            id_map_copy = id_map.copy()
            rgb_shape = tuple(list(id_map.shape) + [3])
            rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
            for i in range(3):
                rgb_map[..., i] = id_map_copy % 256
                id_map_copy //= 256
            return rgb_map
        color = []
        for _ in range(3):
            color.append(id_map % 256)
            id_map //= 256
        return color
    
    ## REGIONS
    
    def build_region_instances(self, out_size, elements, class_dict) -> list[Instance]:
        size = self.get_size()
        instances = []
        for element in elements:
            for element_class, element_coords in self._iter_class_coords(element, class_dict):
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
    
    def build_region_pano(self, out_size, elements, class_dict):
        """
        Create the pano version of the regions
        """
        size = self.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for element in elements:
            for element_class, element_coords in self._iter_class_coords(element, class_dict):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                rgb_color = self.id2rgb(_id)
                cv2.fillPoly(pano_mask, [rounded_coords], rgb_color)
                
                segment: SegmentsInfo = {
                    "id": _id,
                    "category_id": element_class,
                    "iscrowd": False
                }
                segments_info.append(segment)
                
                _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {self.filepath} does not contains regions")
        return pano_mask, segments_info
        
    def build_region_mask(self, out_size, elements, class_dict):
        """
        Builds a "image" mask of desired elements
        """
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for element in elements:
            for element_class, element_coords in self._iter_class_coords(element, class_dict):
                coords = self._scale_coords(element_coords, out_size, size)
                rounded_coords = np.round(coords).astype(np.int32)
                cv2.fillPoly(mask, [rounded_coords], element_class)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains regions")
        return mask
    
    ## BASELINE
    
    def build_baseline_instances(self, out_size, line_width):
        baseline_class = 0
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        instances = []
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            mask.fill(0)
            # HACK Currenty the most simple quickest solution used can probably be optimized
            cv2.polylines(mask, [rounded_coords.reshape(-1, 1, 2)], False, 255, line_width)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                continue
            contour_coords = np.asarray(contours[0]).reshape(-1,2)
            
            bbox = self._bounding_box(coords)
            bbox_mode = structures.BoxMode.XYXY_ABS
            flattened_coords = contour_coords.flatten().tolist()
            instance: Instance = {
                "bbox"        : bbox,
                "bbox_mode"   : bbox_mode,
                "category_id" : baseline_class,
                "segmentation": [flattened_coords],
                "keypoints"   : [],
                "iscrowd"     : False
            }
            instances.append(instance)
            
        return instances
        
    def build_baseline_pano(self, out_size, line_width):
        baseline_class = 0
        size = self.get_size()
        pano_mask = np.zeros((*out_size, 3), np.uint8)
        segments_info = []
        _id = 1
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            rgb_color = self.id2rgb(_id)
            cv2.polylines(pano_mask, [rounded_coords.reshape(-1, 1, 2)], False, rgb_color, line_width)
            segment: SegmentsInfo = {
                "id": _id,
                "category_id": baseline_class,
                "iscrowd": False
            }
            segments_info.append(segment)
            _id += 1
        if not pano_mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return pano_mask, segments_info
    
    def build_baseline_mask(self, out_size, line_width):
        """
        Builds a "image" mask of Baselines on XML-PAGE
        """
        baseline_color = 1
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.polylines(mask, [rounded_coords.reshape(-1, 1, 2)], False, baseline_color, line_width)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return mask
    
    
    ## START
    
    def build_start_mask(self, out_size, line_width):
        """
        Builds a "image" mask of Starts on XML-PAGE
        """
        start_color = 1
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[0]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(mask, rounded_coords, line_width, start_color, -1)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return mask
    
    ## START
    
    def build_end_mask(self, out_size, line_width):
        """
        Builds a "image" mask of Ends on XML-PAGE
        """
        end_color = 1
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)[-1]
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.circle(mask, rounded_coords, line_width, end_color, -1)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return mask
    
    ## SEPARATOR
    
    def build_separator_mask(self, out_size, line_width):
        """
        Builds a "image" mask of Separators on XML-PAGE
        """
        separator_color = 1
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            coords_start = rounded_coords[0]
            cv2.circle(mask, coords_start, line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(mask, coords_end, line_width, separator_color, -1)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return mask
    
    ## BASELINE + SEPARATOR
    
    def build_baseline_separator_mask(self, out_size, line_width):
        """
        Builds a "image" mask of Separators and Baselines on XML-PAGE
        """
        baseline_color = 1
        separator_color = 2
        
        size = self.get_size()
        mask = np.zeros(out_size, np.uint8)
        for baseline_coords in self._iter_baseline_coords():
            coords = self._scale_coords(baseline_coords, out_size, size)
            rounded_coords = np.round(coords).astype(np.int32)
            cv2.polylines(mask, [coords.reshape(-1, 1, 2)], False, baseline_color, line_width)
            
            coords_start = rounded_coords[0]
            cv2.circle(mask, coords_start, line_width, separator_color, -1)
            coords_end = rounded_coords[-1]
            cv2.circle(mask, coords_end, line_width, separator_color, -1)
        if not mask.any():
            self.logger.warning(f"File {self.filepath} does not contains baselines")
        return mask
    
    ## TEXT

    def get_text(self, element):
        """
        get Text defined for element
        """
        text_node = element.find("".join(["./", self.base, "TextEquiv"]))
        if text_node == None:
            self.logger.warning(
                f"No Text node found for line {self.get_id(element)} at {self.name}"
            )
            return ""
        else:
            text_data = text_node.find("*").text
            if text_data == None:
                self.logger.warning(
                    f"No text found in line {self.get_id(element)} at {self.filepath}"
                )
                return ""
            else:
                return text_data.encode("utf-8").strip()

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.root.findall("".join([".//", self.base, "TextRegion"])):
            r_id = self.get_id(element)
            for line in element.findall("".join([".//", self.base, "TextLine"])):
                l_id = self.get_id(line)
                data["_".join([r_id, l_id])] = self.get_text(line)

        return data

    def write_transcriptions(self, out_dir):
        """write out one txt file per text line"""
        # for line, text in self.get_transcription().iteritems():
        for line, text in list(self.get_transcription().items()):
            fh = open(
                os.path.join(out_dir, "".join([self.name, "_", line, ".txt"])), "w"
            )
            fh.write(text + "\n")
            fh.close()
            
    ## NEW PAGEXML

    def new_page(self, name, rows, cols):
        """create a new PAGE xml"""
        self.xml = ET.Element("PcGts")
        self.xml.attrib = self.XMLNS
        metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(metadata, "Creator").text = self.creator
        ET.SubElement(metadata, "Created").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        ET.SubElement(metadata, "LastChange").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        self.page = ET.SubElement(self.xml, "Page")
        self.page.attrib = {
            "imageFilename": name,
            "imageWidth"   : cols,
            "imageHeight"  : rows,
        }

    def add_element(self, r_class, r_id, r_type, r_coords, parent=None):
        """add element to parent node"""
        parent = self.page if parent == None else parent
        t_reg = ET.SubElement(parent, r_class)
        t_reg.attrib = {
            #"id": "_".join([r_class, str(r_id)]),
            "id": str(r_id),
            "custom": "".join(["structure {type:", r_type, ";}"]),
        }
        ET.SubElement(t_reg, "Coords").attrib = {"points": r_coords}
        return t_reg

    def remove_element(self, element, parent=None):
        """remove element from parent node"""
        parent = self.page if parent == None else parent
        parent.remove(element)

    def add_baseline(self, b_coords, parent):
        """add baseline element ot parent line node"""
        ET.SubElement(parent, "Baseline").attrib = {"points": b_coords}

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        tree.write(self.filepath, encoding="UTF-8", xml_declaration=True)

    def _indent(self, elem, level=0):
        """
        Function borrowed from: 
            http://effbot.org/zone/element-lib.htm#prettyprint
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
