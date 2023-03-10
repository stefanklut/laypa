from multiprocessing import Pool
import os
from tqdm import tqdm
import cv2
import argparse
from pathlib import Path
import numpy as np

from utils.input_utils import clean_input_paths, get_file_paths

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine the data from 3 images into 1")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-b", "--baseline", help="Baseline", nargs="+", action="extend", required=True, type=str)
    io_args.add_argument("-s", "--start", help="Start", nargs="+", action="extend", required=True, type=str)
    io_args.add_argument("-e", "--end", help="End", nargs="+", action="extend", required=True, type=str)
    io_args.add_argument("-o", "--output", help="Output file", required=True, type=str)
    
    args = parser.parse_args()
    return args

class CombineStartEnd():
    # TODO Documentation if this remains relevant
    def __init__(self, baseline_image_paths, start_image_paths, end_image_paths, output_path: Path) -> None:
        self.baseline_image_paths = clean_input_paths(baseline_image_paths)
        self.start_image_paths = clean_input_paths(start_image_paths)
        self.end_image_paths = clean_input_paths(end_image_paths)
        
        if not output_path.is_dir():
            print(f"Could not find output dir ({output_path}), creating one at specified location")
            output_path.mkdir(parents=True)
        self.output_path = output_path
        
        self.image_formats = [".png"]
        self.disable_check = False #No argument flag for now
        
    def combine(self, baseline_image_path, start_image_path, end_image_path):
        baseline_image = cv2.imread(str(baseline_image_path), cv2.IMREAD_GRAYSCALE)
        start_image = cv2.imread(str(start_image_path), cv2.IMREAD_GRAYSCALE)
        end_image = cv2.imread(str(end_image_path), cv2.IMREAD_GRAYSCALE)
        
        image = np.stack([end_image, start_image, baseline_image], axis=-1)
        # image = image[..., ::-1] #Flip for BGR
        
        output_image_path = self.output_path.joinpath(baseline_image_path.name)
        # print(output_image_path)
        
        cv2.imwrite(str(output_image_path), image)
    
    def combine_wrapper(self, info):
        baseline_image_path, start_image_path, end_image_path = info
        self.combine(baseline_image_path, start_image_path, end_image_path)
        
    def run(self):
        
        baseline_image_paths = get_file_paths(self.baseline_image_paths, self.image_formats, self.disable_check)
        start_image_paths = get_file_paths(self.start_image_paths, self.image_formats, self.disable_check)
        end_image_paths = get_file_paths(self.end_image_paths, self.image_formats, self.disable_check)
        
        if len(start_image_paths) != len(baseline_image_paths):
            raise ValueError(f"Number of images in {self.start_image_paths} does not match number of images in {self.baseline_image_paths}")
        if len(end_image_paths) != len(baseline_image_paths):
            raise ValueError(f"Number of images in {self.end_image_paths} does not match number of images in {self.baseline_image_paths}")
        
        # Single Thread
        # for i in tqdm(range(len(baseline_image_paths))):
        #     baseline_image_path = baseline_image_paths[i]
        #     start_image_path = start_image_paths[i]
        #     end_image_path = end_image_paths[i]
            
        #     combine(baseline_image_path, start_image_path, end_image_path, output_path)
        
        # Multithread   
        info_iter = list(zip(baseline_image_paths, start_image_paths, end_image_paths))
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(
                self.combine_wrapper, info_iter), total=len(info_iter)))

def main(args):
    """
    Quick program to combine start, end, and baseline predictions. These predictions are saved in the RGB channels of a color image.

    Args:
        args (argparse.Namespace): arguments for where to find the images, and the output dir

    Raises:
        FileNotFoundError: dir for baselines is missing
        FileNotFoundError: dir for start is missing
        FileNotFoundError: dir for end is missing
        FileNotFoundError: no images found in the baseline dir
        ValueError: number of start images does not match the baseline images
        ValueError: number of end images does not match the baseline images
    """
    combiner = CombineStartEnd(args.baseline, args.start, args.end, args.output)    
    combiner.run()
        
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)