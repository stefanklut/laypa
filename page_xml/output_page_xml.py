import argparse
import logging

# from multiprocessing.pool import ThreadPool as Pool
import os
import sys
import uuid
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
import ultralytics
import ultralytics.engine
import ultralytics.engine.predictor
import ultralytics.engine.results
from detectron2.config import CfgNode
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from core.setup import get_git_hash
from data.dataset import classes_to_colors
from page_xml.baseline_extractor import baseline_converter, image_to_baselines
from page_xml.page_xml_editor import Baseline, PageXMLEditor, Region, TextLine
from page_xml.xml_regions import XMLRegions
from utils.copy_utils import copy_mode
from utils.image_utils import save_image_array_to_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name
from utils.tempdir import AtomicFileName


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        parents=[XMLRegions.get_parser()], description="Generate pageXML from label sem_seg and images"
    )

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-s", "--sem_seg", help="Input sem_seg folder/files", nargs="+", default=[], required=True, type=str)
    io_args.add_argument("-i", "--input", help="Input image folder/files", nargs="+", default=[], required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args


class OutputPageXML:
    """
    Class for the generation of the pageXML from class predictions on images
    """

    def __init__(
        self,
        xml_regions: XMLRegions,
        output_dir: Optional[str | Path] = None,
        cfg: Optional[CfgNode] = None,
        whitelist: Optional[Iterable[str]] = None,
        rectangle_regions: Optional[Iterable[str]] = None,
        min_region_size: int = 10,
        external_processing: bool = True,
        external_processing_grayscale: bool = True,
        save_confidence_heatmap: bool = False,
    ) -> None:
        """
        Class for the generation of the pageXML from class predictions on images

        Args:
            xml_regions (XMLRegions): Contains the page xml configurations
            output_dir (str | Path): Path to output dir
            'regions'. Defaults to None.
            cfg (Optional[CfgNode]): Contains the configuration that is used for providence in the pageXML.
            Defaults to None.
            whitelist (Optional[Iterable[str]]): Names of the configuration fields to be used in the pageXML.
            Defaults to None.
            rectangle_regions (Optional[Iterable[str]]): The regions that have to be described with the minimal rectangle,
            that fits them. Defaults to None.
            min_region_size (int): Minimum size a region has to be, to be considered a valid region.
            Defaults to 10 pixels.
            external_processing (bool): If True, the sem_seg is processed externally
            external_processing_grayscale (bool): If True, the sem_seg is output for external processing in grayscale
            save_confidence_heatmap (bool): If True, the confidence heatmap is saved
        """

        self.logger = logging.getLogger(get_logger_name())

        self.xml_regions = xml_regions

        self.output_dir = None
        self.page_dir = None
        self.save_confidence_heatmap = save_confidence_heatmap

        if output_dir is not None:
            self.set_output_dir(output_dir)

        self.cfg = cfg

        self.whitelist = set() if whitelist is None else set(whitelist)
        self.min_region_size = min_region_size
        self.rectangle_regions = set() if rectangle_regions is None else set(rectangle_regions)

        self.external_processing = external_processing
        self.grayscale = external_processing_grayscale
        self.classes_to_colors = classes_to_colors(xml_regions.regions, external_processing_grayscale)

    def set_output_dir(self, output_dir: str | Path):
        """
        Set the output dir for the generated PageXMLs and the images
        The output dir will be created if it does not exist.

        Args:
            output_dir (str | Path): Path to output dir
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)
        self.output_dir = output_dir

        page_dir = self.output_dir.joinpath("page")
        if not page_dir.is_dir():
            self.logger.info(f"Could not find page dir ({page_dir}), creating one at specified location")
            page_dir.mkdir(parents=True)
        self.page_dir = page_dir
        if self.save_confidence_heatmap:
            self.confidence_dir = self.output_dir.joinpath("confidence")

    def set_whitelist(self, whitelist: Iterable[str]):
        """
        Set the whitelist for the configuration fields to be saved in the PageXML.

        Args:
            whitelist (Iterable[str]): Names of the configuration fields to be used in the PageXML.
        """
        self.whitelist = set(whitelist)

    def link_image(self, image_path: Path):
        """
        Symlink image to get the correct output structure

        Args:
            image_path (Path): Path to original image

        Raises:
            TypeError: Output dir has not been set
        """
        if self.output_dir is None:
            raise TypeError("Output dir is None")
        image_output_path = self.output_dir.joinpath(image_path.name)

        copy_mode(image_path, image_output_path, mode="link")

    def generate_image_from_sem_seg(self, sem_seg_classes: np.ndarray, old_height: int, old_width: int) -> np.ndarray:
        """
        Generate image from sem_seg

        Args:
            sem_seg (torch.Tensor): sem_seg as tensor
            image_path (Path): Image path, used for path name
            old_height (int): Height of the original image
            old_width (int): Width of the original image

        Raises:
            TypeError: Output dir has not been set

        Returns:
            np.ndarray: Image as numpy array
        """
        if self.output_dir is None:
            raise TypeError("Output dir is None")

        # If we have only two classes, we can use the binary image
        if len(self.classes_to_colors) == 2:
            return sem_seg_classes * 255

        if self.grayscale:
            image = np.zeros((old_height, old_width), dtype=np.uint8)
        else:
            image = np.zeros((old_height, old_width, 3), dtype=np.uint8)
        for class_id, color in enumerate(self.classes_to_colors):
            # Skip background
            if class_id == 0:
                continue
            image[sem_seg_classes == class_id] = color

        return image

    def add_baselines_to_page(
        self,
        page_xml_editor: PageXMLEditor,
        xml_path: Path,
        sem_seg_tensor: torch.Tensor,
        old_height: int,
        old_width: int,
        upscale: bool = False,
    ) -> PageXMLEditor:
        """
        Add baselines to the pageXML editor.
        The baselines are generated from the sem_seg image.

        Args:
            page_xml_editor (PageXMLEditor): PageXML editor
            xml_path (Path): Path to the PageXML file
            sem_seg (torch.Tensor): sem_seg as tensor
            old_height (int): Height of the original image
            old_width (int): Width of the original image
            upscale (bool, optional): If True, the sem_seg is upscaled to the original image size. Defaults to False.

        Raises:
            ValueError: Page not found in PageXML editor
            ValueError: TextRegion not found in PageXML editor

        Returns:
            PageXMLEditor: PageXML editor with the baselines added
        """
        page = page_xml_editor.find(".//Page")
        if page is None:
            raise ValueError("Page not found in pageXML")
        # image_name = page.attrib["imageFilename"]
        page.append(
            Region.with_tag(
                "TextRegion",
                np.asarray([[0, 0], [0, old_height], [old_width, old_height], [old_width, 0]]),
                id=f"textregion_{uuid.uuid4()}",
            )
        )

        height, width = sem_seg_tensor.shape[-2:]

        height, width = sem_seg_tensor.shape[-2:]

        if upscale:
            scaling = np.asarray([1, 1])
            height, width = old_height, old_width
        else:
            height, width = sem_seg_tensor.shape[-2:]
            scaling = np.asarray([old_width, old_height] / np.asarray([width, height]))

        sem_seg_classes, confidence = self.sem_seg_to_classes_and_confidence(sem_seg_tensor, height, width)

        sem_seg_classes = sem_seg_classes.cpu().numpy().astype(np.uint8)
        mean_confidence = torch.mean(confidence).cpu().numpy().item()

        if self.cfg is not None:
            page_xml_editor.add_processing_step(
                get_git_hash(),
                self.cfg.LAYPA_UUID,
                self.cfg,
                self.whitelist,
                confidence=mean_confidence,
            )

        # Apply a color map
        if self.save_confidence_heatmap:
            self.save_heatmap(confidence, xml_path)

        minimum_width = 15 / scaling[0]
        minimum_height = 3 / scaling[1]
        step = 50 // scaling[0]

        xml_file = str(xml_path)

        coords_baselines = image_to_baselines(
            sem_seg_classes, self.xml_regions, xml_file, minimum_width=minimum_width, minimum_height=minimum_height, step=step
        )

        text_region = page.find("./TextRegion")
        if text_region is None:
            raise ValueError("TextRegion not found in pageXML")
        for coords_baseline in coords_baselines:
            coords_baseline = (coords_baseline * scaling).astype(np.float32)
            bbox = cv2.boundingRect(coords_baseline)
            coords_text_line = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[1] + bbox[3]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    [bbox[0] + bbox[2], bbox[1]],
                ]
            )
            text_line = TextLine(coords_text_line, id=f"textline_{uuid.uuid4()}")
            baseline = Baseline(coords_baseline)
            text_line.append(baseline)
            text_region.append(text_line)

        return page_xml_editor

    def add_regions_to_page(
        self,
        page_xml_editor: PageXMLEditor,
        xml_path: Path,
        sem_seg_tensor: torch.Tensor,
        old_height: int,
        old_width: int,
        upscale: bool = False,
    ) -> PageXMLEditor:
        """
        Add regions to the PageXML editor.
        The regions are generated from the sem_seg image.

        Args:
            page_xml_editor (PageXMLEditor): PageXML editor
            xml_path (Path): Path to the PageXML file
            sem_seg (torch.Tensor): sem_seg as tensor
            old_height (int): Height of the original image
            old_width (int): Width of the original image
            upscale (bool, optional): If True, the sem_seg is upscaled to the original image size. Defaults to False.

        Raises:
            ValueError: Page not found in PageXML editor

        Returns:
            PageXMLEditor: PageXML editor with the regions added
        """
        height, width = sem_seg_tensor.shape[-2:]

        if upscale:
            scaling = np.asarray([1, 1])
            height, width = old_height, old_width
        else:
            height, width = sem_seg_tensor.shape[-2:]
            scaling = np.asarray([old_width, old_height] / np.asarray([width, height]))

        sem_seg_classes, confidence = self.sem_seg_to_classes_and_confidence(sem_seg_tensor, height, width)

        sem_seg_classes = sem_seg_classes.cpu().numpy()
        mean_confidence = torch.mean(confidence).cpu().numpy().item()

        if self.cfg is not None:
            page_xml_editor.add_processing_step(
                get_git_hash(),
                self.cfg.LAYPA_UUID,
                self.cfg,
                self.whitelist,
                confidence=mean_confidence,
            )

        # Apply a color map
        if self.save_confidence_heatmap:
            self.save_heatmap(confidence, xml_path)

        region_id = 0

        page = page_xml_editor.find(".//Page")
        if page is None:
            raise ValueError("Page not found in pageXML")

        for class_id, region in enumerate(self.xml_regions.regions):
            # Skip background
            if class_id == 0:
                continue
            binary_region_mask = np.zeros_like(sem_seg_classes).astype(np.uint8)
            binary_region_mask[sem_seg_classes == class_id] = 1

            region_type = self.xml_regions.region_types[region]

            contours, hierarchy = cv2.findContours(binary_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                # remove small objects
                if cnt.shape[0] < 4:
                    continue
                if cv2.contourArea(cnt) < self.min_region_size:
                    continue

                region_id += 1

                region_coords = ""
                if region in self.rectangle_regions:
                    # find bounding box
                    rect = cv2.minAreaRect(cnt)
                    poly = cv2.boxPoints(rect) * scaling
                else:
                    # soft a bit the region to prevent spikes
                    epsilon = 0.0005 * cv2.arcLength(cnt, True)
                    approx_poly = cv2.approxPolyDP(cnt, epsilon, True)

                    approx_poly = np.round((approx_poly * scaling)).astype(np.int32)

                    poly = approx_poly.reshape(-1, 2)

                for coords in poly:
                    region_coords = region_coords + f" {round(coords[0])},{round(coords[1])}"

                region_coords = region_coords.strip()

                _uuid = uuid.uuid4()
                page.append(
                    Region.with_tag(
                        region_type,
                        poly,
                        region,
                        id=f"region_{_uuid}_{region_id}",
                    )
                )

        return page_xml_editor

    def generate_single_page_yolo(
        self,
        yolo_output: ultralytics.engine.results.Results,
        image_path: Path,
        old_height: Optional[int] = None,
        old_width: Optional[int] = None,
    ):
        """
        Convert a single prediction into a page

        Args:
            yolo_output: yolo output
            image_path (Path): Image path, used for path name
            old_height (Optional[int], optional): height of the original image. Defaults to None.
            old_width (Optional[int], optional): width of the original image. Defaults to None.
        """

        if self.output_dir is None:
            raise TypeError("Output dir is None")
        if self.page_dir is None:
            raise TypeError("Page dir is None")

        xml_output_path = self.page_dir.joinpath(image_path.stem + ".xml")
        if old_height is None or old_width is None:
            height, width = yolo_output.orig_shape
        else:
            height, width = old_height, old_width

        page_xml_editor = PageXMLEditor()
        page = page_xml_editor.add_page(image_path.name, height, width)

        if self.cfg is not None:
            page_xml_editor.add_processing_step(
                get_git_hash(),
                self.cfg.LAYPA_UUID,
                self.cfg,
                self.whitelist,
                confidence=None,
            )

        if yolo_output.boxes is None:
            page_xml_editor.save_xml(xml_output_path)
            return

        if yolo_output.masks is not None:
            relative_contours = [yolo_output.masks.xyn[i] for i in range(yolo_output.masks.shape[0])]
        else:
            relative_bboxes = [yolo_output.boxes.xyxyn[i] for i in range(yolo_output.boxes.shape[0])]
            relative_contours = [
                np.array(
                    [
                        [relative_bbox[0], relative_bbox[1]],
                        [relative_bbox[2], relative_bbox[1]],
                        [relative_bbox[2], relative_bbox[3]],
                        [relative_bbox[0], relative_bbox[3]],
                    ]
                )
                for relative_bbox in relative_bboxes
            ]

        region_id = 0

        for i in range(len(relative_contours)):
            absolute_contour = relative_contours[i] * np.asarray([width, height])

            class_id = int(yolo_output.boxes.cls[i].cpu().numpy())
            confidence = yolo_output.boxes.conf[i].cpu().numpy()

            region = yolo_output.names[class_id]
            region_type = self.xml_regions.region_types[region]

            region_id += 1

            _uuid = uuid.uuid4()
            page.append(
                Region.with_tag(
                    region_type,
                    absolute_contour,
                    region,
                    id=f"region_{_uuid}_{region_id}",
                )
            )

        page_xml_editor.save_xml(xml_output_path)

    @staticmethod
    def scale_to_range(
        tensor: torch.Tensor,
        min_value: float = 0.0,
        max_value: float = 1.0,
        tensor_min: Optional[float] = None,
        tensor_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Scale tensor to a range

        Args:
            image (torch.Tensor): image to be scaled
            min_value (float, optional): minimum value of the range. Defaults to 0.0.
            max_value (float, optional): maximum value of the range. Defaults to 1.0.
            tensor_min (Optional[float], optional): minimum value of the tensor. Defaults to None.
            tensor_max (Optional[float], optional): maximum value of the tensor. Defaults to None.

        Returns:
            torch.Tensor: scaled image
        """

        if tensor_min is None:
            tensor_min = torch.min(tensor).item()
        if tensor_max is None:
            tensor_max = torch.max(tensor).item()

        tensor = (max_value - min_value) * (tensor - tensor_min) / (tensor_max - tensor_min) + min_value

        return tensor

    def save_heatmap(self, scaled_confidence: torch.Tensor, xml_path: Path):
        """
        Save a heatmap of the confidence.

        Args:
            scaled_confidence (torch.Tensor): confidence as tensor.
            confidence_output_path (Path): path to save the heatmap.
        """
        self.confidence_dir.mkdir(parents=True, exist_ok=True)
        confidence_output_path = self.confidence_dir.joinpath(xml_path.stem + "_confidence.png")

        confidence_grayscale = (scaled_confidence * 255).cpu().numpy().astype(np.uint8)
        confidence_colored = cv2.applyColorMap(confidence_grayscale, cv2.COLORMAP_PLASMA)[..., ::-1]
        with AtomicFileName(file_path=confidence_output_path) as path:
            save_image_array_to_path(str(path), confidence_colored)

    def sem_seg_to_classes_and_confidence(
        self,
        sem_seg: torch.Tensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a single prediction into classes and confidence.

        Args:
            sem_seg (torch.Tensor): sem_seg as tensor.

        Returns:
            torch.Tensor, torch.Tensor: classes and confidence.
        """
        sem_seg_normalized = torch.nn.functional.softmax(sem_seg, dim=-3)
        if height is not None and width is not None:
            sem_seg_interpolated = torch.nn.functional.interpolate(
                sem_seg_normalized[None], size=(height, width), mode="bilinear", align_corners=False
            )[0]
        else:
            sem_seg_interpolated = sem_seg_normalized

        confidence, _ = torch.max(sem_seg_normalized, dim=-3)
        sem_seg_classes = torch.argmax(sem_seg_interpolated, dim=-3)

        scaled_confidence = self.scale_to_range(confidence, tensor_min=1 / len(self.xml_regions.regions), tensor_max=1.0)

        return sem_seg_classes, scaled_confidence

    def generate_single_page(
        self,
        sem_seg: torch.Tensor,
        image_path: Path,
        old_height: Optional[int] = None,
        old_width: Optional[int] = None,
    ):
        """
        Convert a single prediction into a page.

        Args:
            sem_seg (torch.Tensor): sem_seg as tensor.
            image_path (Path): Image path, used for path name.
            old_height (Optional[int], optional): height of the original image. Defaults to None.
            old_width (Optional[int], optional): width of the original image. Defaults to None.

        Raises:
            TypeError: Output dir has not been set.
            TypeError: Page dir has not been set.
            NotImplementedError: mode is not known.
        """
        if self.output_dir is None:
            raise TypeError("Output dir is None")
        if self.page_dir is None:
            raise TypeError("Page dir is None")

        if old_height is None or old_width is None:
            old_height, old_width = sem_seg.shape[-2:]

        page_xml_editor = PageXMLEditor()
        page_xml_editor.add_page(image_path.name, old_height, old_width)

        xml_output_path = self.page_dir.joinpath(image_path.stem + ".xml")

        # TODO This external processing is only done for baselines, not for regions. So exlude regions for now
        if self.external_processing and not self.xml_regions.mode == "region":
            sem_seg_output_path = self.page_dir.joinpath(image_path.stem + ".png")
            sem_seg_classes, confidence = self.sem_seg_to_classes_and_confidence(sem_seg, old_height, old_width)
            if self.save_confidence_heatmap:
                self.save_heatmap(confidence, image_path)
            sem_seg_classes = sem_seg_classes.cpu().numpy()
            mean_confidence = torch.mean(confidence).cpu().numpy().item()
            if self.cfg is not None:
                page_xml_editor.add_processing_step(
                    get_git_hash(),
                    self.cfg.LAYPA_UUID,
                    self.cfg,
                    self.whitelist,
                    confidence=mean_confidence,
                )
            colored_image = self.generate_image_from_sem_seg(sem_seg_classes, old_height, old_width)
            with AtomicFileName(file_path=sem_seg_output_path) as path:
                save_image_array_to_path(str(path), colored_image.astype(np.uint8))
            page_xml_editor.save_xml(xml_output_path)
            return

        if self.xml_regions.mode == "region":
            page_xml_editor = self.add_regions_to_page(
                page_xml_editor,
                xml_output_path,
                sem_seg,
                old_height,
                old_width,
            )
            page_xml_editor.save_xml(xml_output_path)
        elif self.xml_regions.mode == "baseline":
            page_xml_editor = self.add_baselines_to_page(
                page_xml_editor,
                xml_output_path,
                sem_seg,
                old_height,
                old_width,
            )
            page_xml_editor.save_xml(xml_output_path)
        else:
            raise NotImplementedError(
                f"No internal processing mode {self.xml_regions.mode}, use external processing by setting external_processing=True"
            )

    def generate_single_page_wrapper(self, info):
        """
        Convert a single prediction into a page.

        Args:
            info (tuple[torch.Tensor | Path, Path]):
                (tuple containing)
                torch.Tensor | Path: mask as array or path to mask.
                Path: original image path.
        """
        mask, image_path = info
        if isinstance(mask, Path):
            mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            mask = torch.as_tensor(mask)
        self.generate_single_page(mask, image_path)

    def run(
        self,
        sem_seg_list: list[torch.Tensor] | list[Path],
        image_path_list: list[Path],
    ) -> None:
        """
        Generate pageXML for all sem_seg-image pairs in the lists.

        Args:
            sem_seg_list (list[np.ndarray] | list[Path]): all sem_seg as arrays or path to the sem_seg.
            image_path_list (list[Path]): path to the original image.

        Raises:
            ValueError: length of sem_seg list and image list do not match.
        """

        if len(sem_seg_list) != len(image_path_list):
            raise ValueError(f"Sem_seg must match image paths in length: {len(sem_seg_list)} v. {len(image_path_list)}")

        # Do not run multiprocessing for single images
        if len(sem_seg_list) == 1:
            self.generate_single_page_wrapper((sem_seg_list[0], image_path_list[0]))
            return

        # #Single thread
        # for sem_seg_i, image_path_i in tqdm(zip(sem_seg_list, image_path_list), total=len(sem_seg_list)):
        #     self.generate_single_page((sem_seg_i, image_path_i))

        # Multi thread
        with Pool(os.cpu_count()) as pool:
            _ = list(
                tqdm(
                    iterable=pool.imap_unordered(self.generate_single_page_wrapper, list(zip(sem_seg_list, image_path_list))),
                    total=len(sem_seg_list),
                    desc="Generating PageXML",
                )
            )


def main(args):
    sem_seg_paths = get_file_paths(args.sem_seg, formats=[".png"])
    image_paths = get_file_paths(args.input, formats=SUPPORTED_IMAGE_FORMATS)

    xml_regions = XMLRegions(
        mode=args.mode,
        regions=args.regions,
        region_type=args.region_types,
        merge_regions=args.merge_regions,
        line_width=args.line_width,
    )

    gen_page = OutputPageXML(
        xml_regions=xml_regions,
        output_dir=args.output,
        rectangle_regions=args.rectangle_regions,
        min_region_size=args.min_region_size,
    )

    gen_page.run(sem_seg_paths, image_paths)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
