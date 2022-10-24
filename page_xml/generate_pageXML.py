import argparse
import logging
from multiprocessing import Pool
import os
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
    parser.add_argument("-i", "--input", help="Input folder/file",
                        required=True, type=str)
    parser.add_argument(
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
        
        
        for i, region in enumerate(self.regions):
            binary_region_mask = np.zeros_like(mask)
            binary_region_mask[mask == i] = 1
            
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
                
                # --- soft a bit the region to prevent spikes
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
            
                approx_poly = np.round((approx_poly * scaling)).astype(np.int32)
        
        page.save_xml()

    
    def run(self, mask_list: list[np.ndarray], image_path_list: list[Path]):
        # #Single thread
        # for mask_i, image_path_i in tqdm(zip(mask_list, image_path_list), total=len(mask_list)):
        #     self.generate_single_page((mask_i, image_path_i))
        
        
        # Multi thread
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(
                self.generate_single_page, list(zip(mask_list, image_path_list))), total=len(mask_list)))

def main():
    
    
if __name__ == "__main__":
    args = get_arguments()
    
    
        