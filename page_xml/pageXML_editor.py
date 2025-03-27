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
    def __init__(
        self,
        git_hash: str,
        uuid: str,
        cfg: CfgNode,
        whitelist: Iterable[str],
        confidence: Optional[float] = None,
        **kwargs,
    ):
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

        if confidence is not None:
            confidence_element = ET.SubElement(labels, "Label")
            confidence_element.attrib = {
                "type": "confidence",
                "value": str(confidence),
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
        super().__init__(element=PcGts(), **kwargs)

    def save_xml(self, filepath: Path):
        """write out XML file of current PAGE data"""
        self._indent(self.getroot())
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


class PageXMLEditor(PageXML):
    """Class to process PAGE xml files"""

    def __init__(self, filepath: Optional[Path | str] = None):
        self.logger = logging.getLogger(get_logger_name())
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

        ET.register_namespace("", self.XMLNS["xmlns"])

        self.filepath = Path(filepath) if filepath is not None else None
        if self.filepath is not None:
            if self.filepath.exists():
                self.parse(self.filepath)
                # HACK Remove namespace from tree
                default_namespace = self.getroot().tag.split("}")[0] + "}"
                for elem in self.iter():
                    name_space = elem.tag.split("}")[0] + "}"
                    if name_space and name_space != default_namespace:
                        raise ValueError(f"Found unexpected namespace {name_space}")
                    if name_space == default_namespace:
                        elem.tag = elem.tag.split("}")[1]
            else:
                raise FileNotFoundError(f"File {filepath} does not exist")

        if filepath is None:
            super().__init__()
            self.add_metadata()

    def set_size(self, size: tuple[int, int]):
        self.size = size

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        if self.size is not None:
            return self.size

        page = self.find(".//Page")
        if page is None:
            raise ValueError("Page element is missing in the XML file.")

        img_width_str = page.get("imageWidth")
        if img_width_str is None:
            raise ValueError("imageWidth attribute is missing in the XML file.")
        img_width = int(img_width_str)
        img_height_str = page.get("imageHeight")
        if img_height_str is None:
            raise ValueError("imageHeight attribute is missing in the XML file.")
        img_height = int(img_height_str)
        self.size = (img_height, img_width)

        return self.size

    def get_region_type(self, element):
        """
        Returns the type of element
        """
        try:
            re_match = re.match(r".*structure {.*type:(.*);.*}", element.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        e_type = re_match.group(1)

        return e_type

    def get_id(self, element) -> str:
        """
        get Id of current element
        """
        return str(element.attrib.get("id"))

    def get_zones(self, region_names):
        to_return = {}
        idx = 0
        for element in region_names:
            for node in self.iterfind(f".//{element}"):
                to_return[idx] = {
                    "coords": self.get_coords(node),
                    "type": self.get_region_type(node),
                    "id": self.get_id(node),
                }
                idx += 1
        if to_return:
            return to_return
        else:
            return None

    def get_coords(self, element: ET.Element) -> np.ndarray:
        coords_element = element.find("Coords")
        if coords_element is None:
            raise ValueError(f"'Coords' element not found in the provided element: {element.tag}")
        points_attr = coords_element.attrib.get("points")
        if points_attr is None:
            raise ValueError(f"'points' attribute is missing in the 'Coords' element: {coords_element.tag}")
        str_coords = points_attr.split()
        return np.array([i.split(",") for i in str_coords]).astype(np.int32)

    def iter_class_coords(self, element, class_dict):
        for node in self.iterfind(f".//{element}"):
            element_type = self.get_region_type(node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            element_coords = self.get_coords(node)

            # Ignore lines
            if element_coords.shape[0] < 3:
                continue

            yield element_class, element_coords

    def iter_baseline_coords(self):
        for node in self.iterfind(".//Baseline"):
            str_coords = node.attrib.get("points")
            # Ignoring empty baselines
            if str_coords is None:
                continue
            split_str_coords = str_coords.split()
            # Ignoring empty baselines
            if len(split_str_coords) == 0:
                continue
            # HACK Doubling single value baselines (otherwise they are not drawn)
            if len(split_str_coords) == 1:
                split_str_coords = split_str_coords * 2  # Double list [value]*2 for cv2.polyline
            coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
            yield coords

    def iter_class_baseline_coords(self, element, class_dict):
        for class_node in self.iterfind(f".//{element}"):
            element_type = self.get_region_type(class_node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            for baseline_node in class_node.iterfind(".//Baseline"):
                str_coords = baseline_node.attrib.get("points")
                # Ignoring empty baselines
                if str_coords is None:
                    continue
                split_str_coords = str_coords.split()
                # Ignoring empty baselines
                if len(split_str_coords) == 0:
                    continue
                # HACK Doubling single value baselines (otherwise they are not drawn)
                if len(split_str_coords) == 1:
                    split_str_coords = split_str_coords * 2  # Double list [value]*2 for cv2.polyline
                coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
                yield element_class, coords

    def iter_text_line_coords(self):
        for node in self.iterfind(".//TextLine"):
            coords = self.get_coords(node)
            yield coords

    def add_metadata(self):
        metadata = Metadata()
        metadata.append(Creator("laypa"))
        metadata.append(Created())
        metadata.append(LastChange())
        self.getroot().append(metadata)
        return metadata

    def add_page(
        self,
        image_filename: str,
        image_height: int,
        image_width: int,
    ):
        page = Page(image_filename, image_width, image_height)
        self.getroot().append(page)
        return page

    def add_processing_step(
        self,
        git_hash: str,
        uuid: str,
        cfg: CfgNode,
        whitelist: Iterable[str],
        confidence: Optional[float] = None,
    ):
        processing_step = LaypaProcessingStep(git_hash, uuid, cfg, whitelist, confidence)
        metadata = self.find("Metadata")
        if metadata is None:
            metadata = self.add_metadata()

        metadata.append(processing_step)
        return processing_step

    def get_text(self, element):
        """
        get Text defined for element
        """
        text_node = element.find("TextEquiv")
        if text_node is None:
            self.logger.info(f"No Text node found for line {self.get_id(element)} at {self.filepath}")
            return ""
        else:
            child_node = text_node.find("*")
            if child_node is None or child_node.text is None:
                self.logger.info(f"No text found in line {self.get_id(element)} at {self.filepath}")
                return ""
            else:
                return child_node.text

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.iterfind(".//TextRegion"):
            r_id = self.get_id(element)
            for line in element.iterfind(".//TextLine"):
                l_id = self.get_id(line)
                data["_".join([r_id, l_id])] = self.get_text(line)

        return data
