import argparse
import logging
from multiprocessing.pool import Pool
# from multiprocessing.pool import ThreadPool as Pool
import os
import sys
from typing import Optional
import numpy as np
from pathlib import Path
import uuid
import torch

from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.tempdir import AtomicFileName
from utils.logging_utils import get_logger_name
from utils.image_utils import save_image_array_to_path
from utils.copy_utils import copy_mode
from utils.input_utils import get_file_paths
from page_xml.xmlPAGE import PageData
from page_xml.xml_regions import XMLRegions

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLRegions.get_parser()],
        description="Generate pageXML from label mask and images")
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-a", "--mask", help="Input mask folder/files", nargs="+", default=[],
                        required=True, type=str)
    io_args.add_argument("-i", "--input", help="Input image folder/files", nargs="+", default=[],
                        required=True, type=str)
    io_args.add_argument(
        "-o", "--output", help="Output folder", required=True, type=str)

    args = parser.parse_args()
    return args

class OutputPageXML(XMLRegions):
    """
    Class for the generation of the pageXML from class predictions on images
    """
    
    def __init__(self, 
                 mode: str,
                 output_dir: Optional[str|Path] = None,
                 line_width: Optional[int] = None,
                 regions: Optional[list[str]] = None,
                 merge_regions: Optional[list[str]] = None,
                 region_type: Optional[list[str]] = None) -> None:
        """
        Class for the generation of the pageXML from class predictions on images

        Args:
            mode (str): mode of the region type
            output_dir (str | Path): path to output dir
            line_width (Optional[int], optional): width of line. Defaults to None.
            regions (Optional[list[str]], optional): list of regions to extract from pageXML. Defaults to None.
            merge_regions (Optional[list[str]], optional): list of region to merge into one. Defaults to None.
            region_type (Optional[list[str]], optional): type of region for each region. Defaults to None.
        """
        super().__init__(mode, line_width, regions, merge_regions, region_type)
        
        self.output_dir = None
        self.page_dir = None
        
        self.logger = logging.getLogger(get_logger_name())
        
        if output_dir is not None:
            self.set_output_dir(output_dir)
        
        self.regions = self.get_regions()
        
    def set_output_dir(self, output_dir: str|Path):
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
        
        copy_mode(image_path, image_output_path, mode="symlink")
    
    def generate_single_page(self, mask: torch.Tensor, image_path: Path, old_height: Optional[int] = None, old_width: Optional[int] = None):
        """
        Convert a single prediction into a page

        Args:
            mask (torch.Tensor): mask as tensor
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
            old_height, old_width = mask.shape[-2:]
        
        height, width = mask.shape[-2:]
        
        scaling = np.asarray([old_width, old_height] / np.asarray([width, height]))
        # scaling = np.asarray((1,1))
        
        page = PageData(xml_output_path)
        page.new_page(image_path.name, str(old_height), str(old_width))
        
        if self.mode == 'region':
            mask = torch.argmax(mask, dim=-3).cpu().numpy()
            
            region_id = 0
            
            for i, region in enumerate(self.regions):
                if region == "background":
                    continue
                binary_region_mask = np.zeros_like(mask).astype(np.uint8)
                binary_region_mask[mask == i] = 1
                
                region_type = self.region_types[region]
                
                contours, hierarchy = cv2.findContours(
                    binary_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                for cnt in contours:
                    # --- remove small objects
                    if cnt.shape[0] < 4:
                        continue
                    # TODO what size
                    # if cv2.contourArea(cnt) < size:
                    #     continue
                    
                    region_id += 1
                    
                    # --- soft a bit the region to prevent spikes
                    epsilon = 0.0005 * cv2.arcLength(cnt, True)
                    approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
                
                    approx_poly = np.round((approx_poly * scaling)).astype(np.int32)
                    
                    region_coords = ""
                    for coords in approx_poly.reshape(-1, 2):
                        region_coords = region_coords + f" {coords[0]},{coords[1]}"
                    region_coords = region_coords.strip()
                    
                    _uuid = uuid.uuid4()
                    text_reg = page.add_element(
                        region_type, f"region_{_uuid}_{region_id}", region, region_coords
                    )
        elif self.mode in ['baseline', 'start', 'end', "separator"]:
            # Push the calculation to outside of the python code <- mask is used by minion
            mask_output_path = self.page_dir.joinpath(image_path.stem + ".png")
            mask = torch.nn.functional.interpolate(
                mask[None], size=(old_height,old_width), mode="bilinear", align_corners=False
            )[0]
            mask = torch.argmax(mask, dim=-3).cpu().numpy()
            with AtomicFileName(file_path=mask_output_path) as path:
                save_image_array_to_path(str(path), (mask * 255).astype(np.uint8))
        elif self.mode in ["baseline_separator"]:
            mask_output_path = self.page_dir.joinpath(image_path.stem + ".png")
            mask = torch.nn.functional.interpolate(
                mask[None], size=(old_height,old_width), mode="bilinear", align_corners=False
            )[0]
            mask = torch.argmax(mask, dim=-3).cpu().numpy()
            with AtomicFileName(file_path=mask_output_path) as path:
                save_image_array_to_path(str(path), (mask * 128).clip(0,255).astype(np.uint8))
        else:
            raise NotImplementedError
                
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
        
    def run(self, 
            mask_list: list[torch.Tensor] | list[Path], 
            image_path_list: list[Path]) -> None:
        """
        Generate pageXML for all mask-image pairs in the lists

        Args:
            mask_list (list[torch.Tensor] | list[Path]): all mask as arrays or path to the mask
            image_path_list (list[Path]): path to the original image

        Raises:
            ValueError: length of mask list and image list do not match
        """
        
        if len(mask_list) != len(image_path_list):
            raise ValueError(f"masks must match image paths in length: {len(mask_list)} v. {len(image_path_list)}")
        
        # Do not run multiprocessing for single images
        if len(mask_list) == 1:
            self.generate_single_page_wrapper((mask_list[0], image_path_list[0]))
            return
        
        # #Single thread
        # for mask_i, image_path_i in tqdm(zip(mask_list, image_path_list), total=len(mask_list)):
        #     self.generate_single_page((mask_i, image_path_i))
        
        # Multi thread
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(iterable=pool.imap_unordered(self.generate_single_page_wrapper, list(zip(mask_list, image_path_list))), 
                          total=len(mask_list),
                          desc="Generating PageXML"))

def main(args):
    # Formats found here: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#imread
    image_formats = [".bmp", ".dib",
                     ".jpeg", ".jpg", ".jpe",
                     ".jp2",
                     ".png",
                     ".webp",
                     ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
                     ".pfm",
                     ".sr", ".ras",
                     ".tiff", ".tif",
                     ".exr",
                     ".hdr", ".pic"]
    mask_paths = get_file_paths(args.mask, formats=[".png"])
    image_paths = get_file_paths(args.input, formats=image_formats)
    
    gen_page = OutputPageXML(mode=args.mode,
                          output_dir=args.output,
                          line_width=args.line_width,
                          regions=args.regions,
                          merge_regions=args.merge_regions,
                          region_type=args.region_type)
    
    gen_page.run(mask_paths, image_paths)
    
    
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)
    
        