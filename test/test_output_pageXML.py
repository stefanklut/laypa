import unittest
import numpy as np
import sys
from pathlib import Path
import tempfile
import torch
import xml.etree.ElementTree as ET
from os import path


sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.output_pageXML import OutputPageXML
from page_xml.xml_regions import XMLRegions


class TestOutputPageXML(unittest.TestCase):
    def test_one_region_type(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [])
        background = (np.full((10, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) <= 5) * 1
        image = (np.full((10, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) > 5) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")

        page = ET.parse(page_path)

        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(coords_elements))
        self.assertEqual("50,0 50,90 90,90 90,0", coords_elements[0].attrib.get("points"))

    def test_multiple_region_types(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo", "Text"], [], ["ImageRegion:Photo", "TextRegion:Text"])
        xml = OutputPageXML(xml_regions, output, None, [])
        background = (np.full((10, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) <= 2) * 1
        text = (np.full((10, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) > 5) * 1
        image = ((text | background) == 0) * 1
        array = np.array([background, image, text])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)

        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements))
        self.assertEqual("20,0 20,90 40,90 40,0", image_coords_elements[0].attrib.get("points"))

        text_coords_elements = page.findall("./page:Page/page:TextRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(text_coords_elements))
        self.assertEqual("50,0 50,90 90,90 90,0", text_coords_elements[0].attrib.get("points"))

    def test_region_not_square(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [])
        background = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        image = np.invert(background == 1) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements))
        self.assertEqual("40,20 20,40 40,60 50,60 70,40 50,20", image_coords_elements[0].attrib.get("points"))

    def test_rectangle_region_does_cotains_4_points(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [], ["Photo"])
        background = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        image = np.invert(background == 1) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements))
        coord_points = image_coords_elements[0].attrib.get("points")
        self.assertEqual(4, coord_points.count(","), f"Contains more then 4 points: '{coord_points}'")

    def test_rectangle_region_does_create_floating_point_coords(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [], ["Photo"])
        background = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        image = np.invert(background == 1) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements))
        coord_points = image_coords_elements[0].attrib.get("points")
        self.assertEqual(0, coord_points.count("."), f"Probably contains floating points: '{coord_points}'")

    def test_only_rectangle_region_one_type(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo", "Text"], [], ["ImageRegion:Photo", "TextRegion:Text"])
        xml = OutputPageXML(xml_regions, output, None, [], ["Photo"])
        background = np.array(
            [
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            ]
        )

        image = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        text = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            ]
        )
        array = np.array([background, image, text])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements))
        image_coord_points = image_coords_elements[0].attrib.get("points")
        self.assertEqual(4, image_coord_points.count(","), f"ImageRegion Contains more then 4 points: '{image_coord_points}'")
        text_coords_elements = page.findall("./page:Page/page:TextRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(text_coords_elements))
        text_coord_points = text_coords_elements[0].attrib.get("points")
        self.assertEqual(6, text_coord_points.count(","), f"TextRegion less than 6 points: '{text_coord_points}'")

    @unittest.skip("Not enough time/priority for implementation")
    def test_merge_overlapping_squares(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [], ["Photo"])
        background = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        image = np.invert(background == 1) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 100, 100)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_coords_elements = page.findall("./page:Page/page:ImageRegion/page:Coords", namespaces=namespaces)
        self.assertEqual(1, len(image_coords_elements), "more than 1 image is found")

    def test_ignores_too_small_regions(self):
        output = tempfile.mktemp("_laypa_test")
        xml_regions = XMLRegions("region", 5, ["Photo"], [], ["ImageRegion:Photo"])
        xml = OutputPageXML(xml_regions, output, None, [], ["Photo"], 10)
        background = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        image = np.invert(background == 1) * 1
        array = np.array([background, image])
        tensor = torch.from_numpy(array)

        xml.generate_single_page(tensor, Path("/tmp/test.png"), 20, 20)

        page_path = path.join(output, "page", "test.xml")
        self.assertTrue(path.exists(page_path), "Page file does not exist")
        page = ET.parse(page_path)
        namespaces = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        image_region_elements = page.findall("./page:Page/page:ImageRegion", namespaces=namespaces)
        self.assertEqual(0, len(image_region_elements))


if __name__ == "__main__":
    unittest.main()
