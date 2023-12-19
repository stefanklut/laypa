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
from detectron2.config import CfgNode
from tqdm import tqdm

from core.setup import get_git_hash

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_regions import XMLRegions
from page_xml.xmlPAGE import PageData
from utils.copy_utils import copy_mode
from utils.image_utils import save_image_array_to_path
from utils.input_utils import get_file_paths, supported_image_formats
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
    ) -> None:
        """
        Class for the generation of the pageXML from class predictions on images

        Args:
            xml_regions (XMLRegions): contains the page xml configurations
            output_dir (str | Path): path to output dir
            'regions'. Defaults to None.
            cfg (Optional[CfgNode]): contains the configuration that is used for providence in the pageXML.
            Defaults to None.
            whitelist (Optional[Iterable[str]]): names of the configuration fields to be used in the pageXML.
            Defaults to None.
            rectangle_regions (Optional[Iterable[str]]): the regions that have to be described with the minimal rectangle,
            that fits them. Defaults to None.
            min_region_size (int): minimum size a region has to be, to be considered a valid region.
            Defaults to 10 pixels.
        """

        self.logger = logging.getLogger(get_logger_name())

        self.xml_regions = xml_regions

        self.output_dir = None
        self.page_dir = None

        if output_dir is not None:
            self.set_output_dir(output_dir)

        self.cfg = cfg

        self.whitelist = set() if whitelist is None else set(whitelist)
        self.min_region_size = min_region_size
        self.rectangle_regions = set() if rectangle_regions is None else set(rectangle_regions)

    def set_output_dir(self, output_dir: str | Path):
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

    def set_whitelist(self, whitelist: Iterable[str]):
        self.whitelist = set(whitelist)

    def link_image(self, image_path: Path):
        """
        Symlink image to get the correct output structure

        Args:
            image_path (Path): path to original image

        Raises:
            TypeError: Output dir has not been set
        """
        if self.output_dir is None:
            raise TypeError("Output dir is None")
        image_output_path = self.output_dir.joinpath(image_path.name)

        copy_mode(image_path, image_output_path, mode="link")

    def generate_single_page(
        self,
        sem_seg: torch.Tensor,
        image_path: Path,
        old_height: Optional[int] = None,
        old_width: Optional[int] = None,
    ):
        """
        Convert a single prediction into a page

        Args:
            sem_seg (torch.Tensor): sem_seg as tensor
            image_path (Path): Image path, used for path name
            old_height (Optional[int], optional): height of the original image. Defaults to None.
            old_width (Optional[int], optional): width of the original image. Defaults to None.

        Raises:
            TypeError: Output dir has not been set
            TypeError: Page dir has not been set
            NotImplementedError: mode is not known
        """
        if self.output_dir is None:
            raise TypeError("Output dir is None")
        if self.page_dir is None:
            raise TypeError("Page dir is None")

        xml_output_path = self.page_dir.joinpath(image_path.stem + ".xml")

        if old_height is None or old_width is None:
            old_height, old_width = sem_seg.shape[-2:]

        height, width = sem_seg.shape[-2:]

        scaling = np.asarray([old_width, old_height] / np.asarray([width, height]))

        page = PageData(xml_output_path)
        page.new_page(image_path.name, str(old_height), str(old_width))

        if self.cfg is not None:
            page.add_processing_step(get_git_hash(), self.cfg.LAYPA_UUID, self.cfg, self.whitelist)

        if self.xml_regions.mode == "region":
            sem_seg = torch.argmax(sem_seg, dim=-3).cpu().numpy()

            region_id = 0

            for region in self.xml_regions.regions:
                if region == "background":
                    continue
                binary_region_mask = np.zeros_like(sem_seg).astype(np.uint8)
                binary_region_mask[sem_seg == self.xml_regions.region_classes[region]] = 1

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
                    text_reg = page.add_element(region_type, f"region_{_uuid}_{region_id}", region, region_coords)
        elif self.xml_regions.mode in ["baseline", "start", "end", "separator"]:
            # Push the calculation to outside of the python code <- mask is used by minion
            sem_seg_output_path = self.page_dir.joinpath(image_path.stem + ".png")
            sem_seg = torch.nn.functional.interpolate(
                sem_seg[None], size=(old_height, old_width), mode="bilinear", align_corners=False
            )[0]
            sem_seg_image = torch.argmax(sem_seg, dim=-3).cpu().numpy()
            with AtomicFileName(file_path=sem_seg_output_path) as path:
                save_image_array_to_path(str(path), (sem_seg_image * 255).astype(np.uint8))
        elif self.xml_regions.mode in ["baseline_separator", "top_bottom"]:
            sem_seg_output_path = self.page_dir.joinpath(image_path.stem + ".png")
            sem_seg = torch.nn.functional.interpolate(
                sem_seg[None], size=(old_height, old_width), mode="bilinear", align_corners=False
            )[0]
            sem_seg_image = torch.argmax(sem_seg, dim=-3).cpu().numpy()
            with AtomicFileName(file_path=sem_seg_output_path) as path:
                save_image_array_to_path(str(path), (sem_seg_image * 128).clip(0, 255).astype(np.uint8))
        else:
            raise NotImplementedError(f"Mode {self.xml_regions.mode} not implemented")

        # TODO Overwrite when multiple image have the same name but different extension
        page.save_xml()

    def generate_single_page_wrapper(self, info):
        """
        Convert a single prediction into a page

        Args:
            info (tuple[torch.Tensor | Path, Path]):
                (tuple containing)
                torch.Tensor | Path: mask as array or path to mask
                Path: original image path
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
        Generate pageXML for all sem_seg-image pairs in the lists

        Args:
            sem_seg_list (list[np.ndarray] | list[Path]): all sem_seg as arrays or path to the sem_seg
            image_path_list (list[Path]): path to the original image

        Raises:
            ValueError: length of sem_seg list and image list do not match
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
    image_paths = get_file_paths(args.input, formats=supported_image_formats)

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
