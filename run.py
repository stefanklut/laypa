import argparse
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Optional
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import datasets.dataset as dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from main import setup_cfg
import torch.nn.functional as F
import torch
from tqdm import tqdm

from natsort import os_sorted

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run file to inference using the model found in the config file")
    
    detectron2_args = parser.add_argument_group("detectron2")
    
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder", type=str)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)
    
    args = parser.parse_args()
    
    return args

class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.TEST.WEIGHTS)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    def __call__(self, original_image):
        return super().__call__(original_image)

class SavePredictor(Predictor):
    def __init__(self, cfg, input_dir, output_dir):
        super().__init__(cfg)
        
        self.input_dir: Optional[Path] = None
        if input_dir is not None:
            self.set_input_dir(input_dir)
        
        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)
            
        # Formats found here: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#imread
        self.image_formats = [".bmp", ".dib",
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
        
    def set_input_dir(self, input_dir: str | Path) -> None:
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input dir ({input_dir}) is not found")

        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"Input path ({input_dir}) is not a directory")

        if not os.access(path=input_dir, mode=os.R_OK):
            raise PermissionError(
                f"No access to {input_dir} for read operations")

        # page_dir = input_dir.joinpath("page")
        # if not input_dir.joinpath("page").exists():
        #     raise FileNotFoundError(f"Sub page dir ({page_dir}) is not found")

        # if not os.access(path=page_dir, mode=os.R_OK):
        #     raise PermissionError(
        #         f"No access to {page_dir} for read operations")

        self.input_dir = input_dir.resolve()
        
    def set_output_dir(self, output_dir: str | Path) -> None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            print(
                f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()
    
    def save_prediction(self, input_path: Path | str):
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        
        if isinstance(input_path, str):
            input_path = Path(input_path)
        image = cv2.imread(str(input_path))
        outputs = super().__call__(image)
        output_image = torch.argmax(outputs["sem_seg"], dim=-3).cpu().numpy()
        output_path = self.output_dir.joinpath(input_path.stem + '.png')
        
        cv2.imwrite(str(output_path), output_image)
        
        return output_path
    
    def process(self):
        if self.input_dir is None:
            raise ValueError("Cannot run when the input dir is not set")
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        
        image_paths = os_sorted([image_path.resolve() for image_path in self.input_dir.glob("*")
                                 if image_path.suffix in self.image_formats])
        # Single thread
        for inputs in tqdm(image_paths):
            self.save_prediction(inputs)
        
        # Multithread <- does not work with cuda
        # with Pool(os.cpu_count()) as pool:
        #     results = list(tqdm(pool.imap_unordered(
        #         self.save_prediction, image_paths), total=len(image_paths)))
    
def main(args) -> None:
    cfg = setup_cfg(args, save_config=False)
    
    predictor = SavePredictor(cfg=cfg, input_dir=args.input, output_dir=args.output)
    
    predictor.process()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
