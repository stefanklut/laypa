# Modified from P2PaLA

import datetime
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import NoneType
from typing import Iterable, Optional, TypedDict

import numpy as np
from detectron2.config import CfgNode

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name
from utils.tempdir import AtomicFileName

_VALID_TYPES = {tuple, list, str, int, float, bool, NoneType}


def convert_to_dict(cfg_node, key_list: list = []):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class Coords(ET.Element):
    def __init__(self, points: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.tag = "Coords"
        self.points = points

    @property
    def points(self) -> np.ndarray:
        str_points = self.attrib["points"]
        points = np.array([i.split(",") for i in str_points]).astype(np.int32)
        return points

    @points.setter
    def points(self, value: np.ndarray):
        assert value.shape[1] == 2, f"Expected 2D array, got {value.shape}"
        str_coords = ""
        for coords in value:
            str_coords += f" {round(coords[0])},{round(coords[1])}"
        self.attrib["points"] = str_coords.strip()


class Baseline(Coords):
    def __init__(self, points: np.ndarray, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "Baseline"


class _Polygon(ET.Element):
    def __init__(self, points: np.ndarray, id: Optional[str] = None, custom: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.tag = "_Polygon"
        self.append(Coords(points))
        if id is not None:
            self.attrib["id"] = id
        if custom is not None:
            self.attrib["custom"] = custom


class TextLine(_Polygon):
    def __init__(self, points: np.ndarray, reading_order: Optional[int]=None, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "TextLine"
        self.reading_order = reading_order
        
        self.logger = logging.getLogger(get_logger_name())

    @property
    def reading_order(self) -> Optional[int]:
        try:
            re_match = re.match(r".*readingOrder {index:(\d+);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No reading order defined for {self.attrib["id"]}")
            return None
        if re_match is None:
            self.logger.warning(f"No reading order defined for {self.attrib["id"]}")
            return None
        reading_order_index = re_match.group(1)

        return int(reading_order_index)

    @reading_order.setter
    def reading_order(self, value: Optional[int]):
        if value is not None:
            self.attrib["custom"] = f"readingOrder {{index:{value};}}"


class TextEquiv(ET.Element):
    def __init__(self, value: str, **kwargs):
        super().__init__(**kwargs)
        self.tag = "TextEquiv"
        unicode = ET.SubElement(self, "Unicode")
        unicode.text = value


class Region(_Polygon):
    def __init__(self, points: np.ndarray, region_type: str, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "Region"
        self.attrib["custom"] = f"structure {{type:{region_type};}}"
        
        self.logger = logging.getLogger(get_logger_name())
        
    @property
    def region_type(self) -> Optional[str]:
        try:
            re_match = re.match(r".*structure {.*type:(.*);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.attrib["id"]}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.attrib["id"]}")
            return None
        region_type = re_match.group(1)

        return region_type
    
    @region_type.setter
    def region_type(self, value: str):
        if value is not None:
            self.attrib["custom"] = f"structure {{type:{value};}}"


class PcGts(ET.Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tag = "PcGts"
        self.attrib = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
        }


class Metadata(ET.Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tag = "Metadata"
        creator = ET.SubElement(self, "Creator")
        creator.text = "Laypa"
        created = ET.SubElement(self, "Created")
        created.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        last_change = ET.SubElement(self, "LastChange")
        last_change.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        
class Page(ET.Element):
    def __init__(self, imageFilename: str, imageWidth: int, imageHeight: int, **kwargs):
        super().__init__(**kwargs)
        self.tag = "Page"
        self.attrib = {
            "imageFilename": imageFilename,
            "imageWidth": str(imageWidth),
            "imageHeight": str(imageHeight),
        }


class PageXML(ET.ElementTree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._root = PcGts()

    def save_xml(self, filepath: Path):
        """write out XML file of current PAGE data"""
        self._indent(self._root)
        tree = ET.ElementTree(self._root)
        with AtomicFileName(filepath) as path:
            tree.write(path, encoding="UTF-8", xml_declaration=True)

    def _indent(self, elem, level=0):
        """
        Function borrowed from:
            http://effbot.org/zone/element-lib.htm
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


class PageXMLCreator:
    """Class to process PAGE xml files"""

    def __init__(self, filepath: Path, creator=None):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.logger = logging.getLogger(get_logger_name())
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

    def set_size(self, size: tuple[int, int]):
        self.size = size

    def new_page(self, name, rows, cols):
        """create a new PAGE xml"""
        self.xml = ET.Element("PcGts")
        self.xml.attrib = self.XMLNS
        self.metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(self.metadata, "Creator").text = self.creator
        ET.SubElement(self.metadata, "Created").text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        ET.SubElement(self.metadata, "LastChange").text = datetime.datetime.today().strftime("%Y-%m-%dT%X")
        self.page = ET.SubElement(self.xml, "Page")
        self.page.attrib = {
            "imageFilename": name,
            "imageWidth": cols,
            "imageHeight": rows,
        }

    def add_processing_step(self, git_hash: str, uuid: str, cfg: CfgNode, whitelist: Iterable[str]):
        if git_hash is None:
            raise TypeError(f"git_hash is None")
        if uuid is None:
            raise TypeError(f"uuid is None")
        if cfg is None:
            raise TypeError(f"cfg is None")
        if whitelist is None:
            raise TypeError(f"whitelist is None")
        if self.metadata is None:
            raise TypeError(f"self.metadata is None")

        processing_step = ET.SubElement(self.metadata, "MetadataItem")
        processing_step.attrib = {
            "type": "processingStep",
            "name": "layout-analysis",
            "value": "laypa",
        }
        labels = ET.SubElement(processing_step, "Labels")
        git_hash_element = ET.SubElement(labels, "Label")
        git_hash_element.attrib = {
            "type": "githash",
            "value": git_hash,
        }

        uuid_element = ET.SubElement(labels, "Label")
        uuid_element.attrib = {
            "type": "uuid",
            "value": uuid,
        }

        for key in whitelist:
            sub_node = cfg
            for sub_key in key.split("."):
                try:
                    sub_node = sub_node[sub_key]
                except KeyError as error:
                    self.logger.error(f"No key {key} in config, missing sub key {sub_key}")
                    raise error
            whilelisted_element = ET.SubElement(labels, "Label")
            whilelisted_element.attrib = {
                "type": key,
                "value": str(convert_to_dict(sub_node)),
            }

    def add_region(self, region_class, region_id, region_type, region_coords, parent=None):
        """add element to parent node"""
        parent = self.page if parent == None else parent
        t_reg = ET.SubElement(parent, region_class)
        t_reg.attrib = {
            "id": str(region_id),
            "custom": f"structure {{type:{region_type};}}",
        }
        ET.SubElement(t_reg, "Coords").attrib = {"points": region_coords}
        return t_reg

    def remove_element(self, element, parent=None):
        """remove element from parent node"""
        parent = self.page if parent == None else parent
        parent.remove(element)

    def add_textline(self, t_coords, t_id, reading_order, parent):
        """add textline element to parent region node"""
        ET.SubElement(parent, "TextLine").attrib = {
            "id": t_id,
            "custom": f"readingOrder {{index:{reading_order};}}",
        }
        ET.SubElement(parent, "Coords").attrib = {"points": t_coords}

    def add_baseline(self, b_coords, parent):
        """add baseline element ot parent line node"""
        ET.SubElement(parent, "Baseline").attrib = {"points": b_coords}

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

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        with AtomicFileName(self.filepath) as path:
            tree.write(path, encoding="UTF-8", xml_declaration=True)
