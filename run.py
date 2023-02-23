import argparse
from multiprocessing import Pool
import os
from pathlib import Path
from typing import Optional, Sequence
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from datasets.augmentations import ResizeShortestEdge
import cv2
from main import setup_cfg
import torch
from tqdm import tqdm

from natsort import os_sorted

from page_xml.generate_pageXML import GenPageXML

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file to inference using the model found in the config file")
    
    detectron2_args = parser.add_argument_group("detectron2")
    
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])
    
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", nargs="+", help="Input folder", type=str, action='extend', required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)
    
    args = parser.parse_args()
    
    return args

class Predictor(DefaultPredictor):
    """
    Predictor runs the model specified in the config, on call the image is processed and the results dict is output
    """
    def __init__(self, cfg):
        """
        Predictor runs the model specified in the config, on call the image is processed and the results dict is output

        Args:
            cfg (CfgNode): config
        """
        super().__init__(cfg)
        
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.TEST.WEIGHTS)
        
        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    def __call__(self, original_image):
        """
        Not really useful, but shows what call needs to be made
        """
        return super().__call__(original_image)

class SavePredictor(Predictor):
    """
    Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML
    """
    def __init__(self, 
                 cfg, 
                 input_paths: str | Path, 
                 output_dir: str | Path, 
                 gen_page: GenPageXML):
        """
        Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML

        Args:
            cfg (CfgNode): config
            input_dir (str | Path): path to input dir
            output_dir (str | Path): path to output dir
            gen_page (GenPageXML): class to convert from predictions to pageXML
        """
        super().__init__(cfg)
        
        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)
        
        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)
            
        if not isinstance(gen_page, GenPageXML):
            raise ValueError(f"Must provide conversion from mask to pageXML. Current type is {type(gen_page)}, not GenPageXML")
            
        self.gen_page = gen_page
            
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
        
    def set_input_paths(self, input_paths: str | Path | Sequence[str|Path]) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        input_paths = self.clean_input_paths(input_paths)
        
        all_input_paths = []

        for input_path in input_paths:
            if not input_path.exists():
                raise FileNotFoundError(f"Input ({input_path}) is not found")

            if not os.access(path=input_path, mode=os.R_OK):
                raise PermissionError(
                    f"No access to {input_path} for read operations")
            
            input_path = input_path.resolve()
            all_input_paths.append(input_path)

        self.input_paths = all_input_paths
        
    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter for the output dir

        Args:
            output_dir (str | Path): path to output dir
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            print(
                f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()
    
    def save_prediction(self, input_path: Path | str):
        """
        Run the model and get the prediction, and save pageXML or a mask image depending on the mode

        Args:
            input_path (Path | str): path to single image

        Raises:
            ValueError: no output dir is specified
        """
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        
        if isinstance(input_path, str):
            input_path = Path(input_path)
        image = cv2.imread(str(input_path))
        outputs = super().__call__(image)
        output_image = torch.argmax(outputs["sem_seg"], dim=-3).cpu().numpy()
        
        self.gen_page.link_image(input_path)
        self.gen_page.generate_single_page(output_image, input_path)
    
    # TODO put in utils together with preprocess
    @staticmethod
    def clean_input_paths(input_paths: str | Path | Sequence[str|Path]) -> Sequence[Path]:
            """
            Make all types of input path conform to list of paths
            
            Args:
                input_paths (str | Path | Sequence[str | Path]): path(s) to dir with images/file(s) with the location of images

            Raises:
                ValueError: Must provide input path
                NotImplementedError: given input paths are the wrong class

            Returns:
                list[Path]: output paths of images
            """
            if not input_paths:
                raise ValueError("Must provide input path")
            
            if isinstance(input_paths, str):
                output = [Path(input_paths)]
            elif isinstance(input_paths, Path):
                output = [input_paths]
            elif isinstance(input_paths, Sequence):
                output = []
                for path in input_paths:
                    if isinstance(path, str):
                        output.append(Path(path))
                    elif isinstance(path, Path):
                        output.append(path)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
            
            return output

    
    def get_file_paths(self, input_paths: str | Path | Sequence[str|Path], disable_check=False) -> Sequence[Path]:
            """
            Get all image paths used for the running the 

            Raises:
                FileNotFoundError: input path not found on the filesystem
                PermissionError: input path not accessible
                FileNotFoundError: dir does not contain any image files 
                FileNotFoundError: image from txt file does not exist
                ValueError: specified path is not a dir or txt file

            Returns:
                list[Path]: image paths
                list[Path]: pageXML paths
            """
            if input_paths is None:
                raise ValueError("Cannot run when the input path is not set")
            
            input_paths = self.clean_input_paths(input_paths)
                
            image_paths = []
            
            for input_path in input_paths:
                if not input_path.exists():
                    raise FileNotFoundError(f"Input dir/file ({input}) is not found")
                
                if not os.access(path=input_path, mode=os.R_OK):
                    raise PermissionError(
                        f"No access to {input_path} for read operations")
                    
                if input_path.is_dir():
                    sub_image_paths = [image_path.absolute() for image_path in input_path.glob("*") if image_path.suffix in self.image_formats]
                    
                    if not disable_check:
                        if len(sub_image_paths) == 0:
                            raise FileNotFoundError(f"No image files found in the provided dir(s)/file(s)")
                        
                elif input_path.is_file() and input_path.suffix == ".txt":
                    with input_path.open(mode="r") as f:
                        sub_image_paths = [Path(line).absolute() for line in f.read().splitlines()]
                        
                    if not disable_check:
                        for path in sub_image_paths:
                            if not path.is_file():
                                raise FileNotFoundError(f"Missing file from the txt file: {input_path}")
                else:
                    raise ValueError(f"Invalid file type: {input_path.suffix}")

                image_paths.extend(sub_image_paths)
            
            return image_paths

    def process(self):
        """
        Run the model on all images within the input dir

        Raises:
            ValueError: no input dir is specified
            ValueError: no output dir is specified
        """
        if self.input_paths is None:
            raise ValueError("Cannot run when the input dir is not set")
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        
        input_path = self.get_file_paths(self.input_paths)
        # Single thread
        for inputs in tqdm():
            self.save_prediction(inputs)
        
        # Multithread <- does not work with cuda
        # with Pool(os.cpu_count()) as pool:
        #     results = list(tqdm(pool.imap_unordered(
        #         self.save_prediction, image_paths), total=len(image_paths)))
    
    
def main(args) -> None:
    cfg = setup_cfg(args, save_config=False)
    
    gen_page = GenPageXML(mode=cfg.MODEL.MODE,
                          output_dir=args.output,
                          line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
                          line_color=cfg.PREPROCESS.BASELINE.LINE_COLOR,
                          regions=cfg.PREPROCESS.REGION.REGIONS,
                          merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
                          region_type=cfg.PREPROCESS.REGION.REGION_TYPE)
    
    predictor = SavePredictor(cfg=cfg, input_paths=args.input, output_dir=args.output, gen_page=gen_page)
    
    predictor.process()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
