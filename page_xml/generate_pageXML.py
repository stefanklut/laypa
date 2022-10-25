import argparse
import logging
from multiprocessing import Pool
import os
import string
import sys
from typing import Optional
import numpy as np
from pathlib import Path
from .xmlPAGE import PageData
from xml_regions import XMLRegions
from tqdm import tqdm
import cv2

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.copy import copy_mode

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[XMLRegions.get_parser()],
        description="Preprocessing an annotated dataset of documents with pageXML")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file",
                        required=True, type=str)
    io_args.add_argument(
        "-o", "--output", help="Output folder", required=True, type=str)
    
    parser.add_argument("-m", "--mode", help="Output mode",
                        choices=["baseline", "region", "both"], default="baseline", type=str)

    parser.add_argument("-w", "--line_width",
                        help="Used line width", type=int, default=5)
    parser.add_argument("-c", "--line_color", help="Used line color",
                        choices=list(range(256)), type=int, metavar="{0-255}", default=1)

    args = parser.parse_args()
    return args

class GenPage(XMLRegions):
    
    def __init__(self, output_dir: str|Path,
                 mode: str,
                 line_width: Optional[int] = None,
                 line_color: Optional[int] = None,
                 regions: Optional[list[str]] = None,
                 merge_regions: Optional[list[str]] = None,
                 region_type: Optional[list[str]] = None) -> None:
        super().__init__(mode, line_width, line_color, regions, merge_regions, region_type)
        
        self.logger = logging.getLogger(__name__)
        
        self.valid_uuid_values = string.ascii_uppercase + string.ascii_lowercase + string.digits
        
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        
        self.regions = self.get_regions()
    
    def generate_single_page(self, info: tuple[np.ndarray, Path]):
        mask, image_path = info
        
        xml_output_path = self.output_dir.joinpath("page", image_path.stem + ".xml")
        image_output_path = self.output_dir.joinpath(image_path.name)
        
        copy_mode(image_path, image_output_path, mode="symlink")
        
        old_height, old_width, channels = cv2.imread(str(image_output_path)).shape
        
        height, width = mask.shape
        
        scaling = np.asarray([width, height]) / np.asarray([old_width, old_height])
        
        page = PageData(xml_output_path, logger=self.logger)
        page.new_page(image_output_path.name, str(old_height), str(old_width))
        
        region_id = 0
        
        for i, region in enumerate(self.regions):
            binary_region_mask = np.zeros_like(mask)
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
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            
                approx_poly = np.round((approx_poly * scaling)).astype(np.int32)
                
                region_coords = ""
                for coords in approx_poly.reshape(-1, 2):
                    region_coords = region_coords + f" {coords[0]}, {coords[1]}"
                region_coords = region_coords.strip()
                
                uuid = str(np.random.choice(self.valid_uuid_values) for _ in range(4))
                # TODO start with region_ actual UUID
                text_reg = page.add_element(
                    region_type, f"r{uuid}_{region_id}", region, region_coords
                )
                
        page.save_xml()

    
    def run(self, mask_list: list[np.ndarray], image_path_list: list[Path]):
        # #Single thread
        # for mask_i, image_path_i in tqdm(zip(mask_list, image_path_list), total=len(mask_list)):
        #     self.generate_single_page((mask_i, image_path_i))
        
        
        # Multi thread
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(
                self.generate_single_page, list(zip(mask_list, image_path_list))), total=len(mask_list)))

def main(args):
    # TODO This
    pass
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)
    
        