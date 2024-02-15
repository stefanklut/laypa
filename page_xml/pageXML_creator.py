import datetime
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import NoneType
from typing import Iterable, Optional

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
        super().__init__("Coords", **kwargs)
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
        super().__init__("_Polygon", **kwargs)
        self.append(Coords(points))
        if id is not None:
            self.attrib["id"] = id
        if custom is not None:
            self.attrib["custom"] = custom


class TextLine(_Polygon):
    def __init__(self, points: np.ndarray, reading_order: Optional[int] = None, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "TextLine"
        self.logger = logging.getLogger(get_logger_name())
        self.reading_order = reading_order

    @property
    def reading_order(self) -> Optional[int]:
        try:
            re_match = re.match(r".*readingOrder {index:(\d+);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No reading order defined for {self.attrib['id']}")
            return None
        if re_match is None:
            self.logger.warning(f"No reading order defined for {self.attrib['id']}")
            return None
        reading_order_index = re_match.group(1)

        return int(reading_order_index)

    @reading_order.setter
    def reading_order(self, value: Optional[int]):
        if value is not None:
            self.attrib["custom"] = f"readingOrder {{index:{value};}}"


class TextEquiv(ET.Element):
    def __init__(self, value: str, **kwargs):
        super().__init__("TextEquiv", **kwargs)
        unicode = ET.SubElement(self, "Unicode")
        unicode.text = value


class Region(_Polygon):
    def __init__(self, points: np.ndarray, region_type: Optional[str] = None, **kwargs):
        super().__init__(points, **kwargs)
        self.logger = logging.getLogger(get_logger_name())

        self.tag = "Region"
        self.region_type = region_type

    @property
    def region_type(self) -> Optional[str]:
        try:
            re_match = re.match(r".*structure {.*type:(.*);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.attrib['id']}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.attrib['id']}")
            return None
        region_type = re_match.group(1)

        return region_type

    @region_type.setter
    def region_type(self, value: Optional[str]):
        if value is not None:
            self.attrib["custom"] = f"structure {{type:{value};}}"

    @classmethod
    def with_tag(cls, tag: str, points: np.ndarray, region_type: Optional[str] = None, **kwargs):
        region = cls(points, region_type, **kwargs)
        region.tag = tag
        return region


class PcGts(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("PcGts", **kwargs)
        self.attrib = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
        }


class Metadata(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("Metadata", **kwargs)


class Creator(ET.Element):
    def __init__(self, text, **kwargs):
        super().__init__("Creator", **kwargs)
        self.text = text


class Created(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("Created", **kwargs)
        self.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")


class LastChange(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("LastChange", **kwargs)
        self.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")


class Page(ET.Element):
    def __init__(self, image_filename: str, image_width: int, image_height: int, **kwargs):
        super().__init__("Page", **kwargs)
        self.attrib = {
            "imageFilename": image_filename,
            "imageWidth": str(image_width),
            "imageHeight": str(image_height),
        }


class LaypaProcessingStep(ET.Element):
    def __init__(self, git_hash: str, uuid: str, cfg: CfgNode, whitelist: Iterable[str], **kwargs):
        super().__init__("MetadataItem", **kwargs)
        self.logger = logging.getLogger(get_logger_name())
        self.attrib = {
            "type": "processingStep",
            "name": "layout-analysis",
            "value": "laypa",
        }
        labels = ET.SubElement(self, "Labels")
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


class PageXML(ET.ElementTree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._root = PcGts()

    def save_xml(self, filepath: Path):
        """write out XML file of current PAGE data"""
        self._indent(self._root)
        with AtomicFileName(filepath) as path:
            super().write(path, encoding="UTF-8", xml_declaration=True)

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

    def __init__(self, filepath: Optional[Path] = None):
        self.pageXML = PageXML()
        self.filepath = filepath
        if filepath is not None:
            if filepath.exists():
                self.pageXML.parse(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} does not exist")

        if filepath is None:
            self.add_metadata()

    def add_metadata(self):
        metadata = Metadata()
        metadata.append(Creator("laypa"))
        metadata.append(Created())
        metadata.append(LastChange())
        self.pageXML.getroot().append(metadata)
        return metadata

    def add_page(self, image_filename: str, image_width: int, image_height: int):
        page = Page(image_filename, image_width, image_height)
        self.pageXML.getroot().append(page)
        return page

    def add_processing_step(self, git_hash: str, uuid: str, cfg: CfgNode, whitelist: Iterable[str]):
        processing_step = LaypaProcessingStep(git_hash, uuid, cfg, whitelist)
        metadata = self.pageXML.getroot().find("Metadata")
        if metadata is None:
            metadata = self.add_metadata()

        metadata.append(processing_step)
        return processing_step
