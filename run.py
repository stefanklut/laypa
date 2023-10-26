import argparse
import logging
import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm

from core.setup import setup_cfg, setup_logging
from datasets.augmentations import ResizeLongestEdge, ResizeScaling, ResizeShortestEdge
from page_xml.output_pageXML import OutputPageXML
from utils.image_utils import load_image_array_from_path, load_image_tensor_from_path
from utils.input_utils import clean_input_paths, get_file_paths
from utils.logging_utils import get_logger_name


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file to inference using the model found in the config file")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", nargs="+", help="Input folder", type=str, action="extend", required=True)
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
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.TEST.WEIGHTS)

        if cfg.INPUT.RESIZE_MODE == "none":
            self.aug = ResizeScaling(scale=1)  # HACK percentage of 1 is no scaling
        elif cfg.INPUT.RESIZE_MODE in ["shortest_edge", "longest_edge"]:
            if cfg.INPUT.RESIZE_MODE == "shortest_edge":
                self.aug = ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
            elif cfg.INPUT.RESIZE_MODE == "longest_edge":
                self.aug = ResizeLongestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
        elif cfg.INPUT.RESIZE_MODE == "scaling":
            self.aug = ResizeScaling(cfg.INPUT.SCALING, cfg.INPUT.MAX_SIZE_TEST)
        else:
            raise NotImplementedError(f"{cfg.INPUT.RESIZE_MODE} is not a known resize mode")

    def gpu_call(self, original_image: torch.Tensor):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            channels, height, width = original_image.shape
            assert channels == 3, f"Must be a BGR image, found {channels} channels"
            image = torch.as_tensor(original_image, dtype=torch.float32, device=self.cfg.MODEL.DEVICE)
            if self.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                image = image[[2, 1, 0], :, :]

            if self.cfg.INPUT.RESIZE_MODE == "none":
                new_height, new_width = height, width
            elif self.cfg.INPUT.RESIZE_MODE in ["shortest_edge", "longest_edge"]:
                new_height, new_width = self.aug.get_output_shape(
                    height, width, self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST
                )
            elif self.cfg.INPUT.RESIZE_MODE == "scaling":
                new_height, new_width = self.aug.get_output_shape(
                    height, width, self.cfg.INPUT.SCALING, self.cfg.INPUT.MAX_SIZE_TEST
                )
            else:
                raise NotImplementedError(f"{self.cfg.INPUT.RESIZE_MODE} is not a known resize mode")

            if self.cfg.INPUT.RESIZE_MODE != "none":
                image = torch.nn.functional.interpolate(image[None], mode="bilinear", size=(new_height, new_width))[0]

            inputs = {"image": image, "height": new_height, "width": new_width}
            predictions = self.model([inputs])[0]
            return predictions, height, width

    def cpu_call(self, original_image: np.ndarray):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            height, width, channels = original_image.shape
            assert channels == 3, f"Must be a BGR image, found {channels} channels"
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image, dtype=torch.float32, device=self.cfg.MODEL.DEVICE).permute(2, 0, 1)
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[[2, 1, 0], :, :]

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

        return predictions, height, width

    def __call__(self, original_image):
        """
        Not really useful, but shows what call needs to be made
        """
        return self.gpu_call(original_image)


class LoadingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        # TODO Move resize and load to this part of the dataloader
        return load_image_tensor_from_path(path), path


def collate_numpy(batch):
    collate_map = default_collate_fn_map

    def new_map(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        return batch

    collate_map.update({np.ndarray: new_map, type(None): new_map})
    return collate(batch, collate_fn_map=collate_map)


class SavePredictor(Predictor):
    """
    Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML
    """

    def __init__(
        self,
        cfg,
        input_paths: str | Path,
        output_dir: str | Path,
        output_page: OutputPageXML,
    ):
        """
        Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML

        Args:
            cfg (CfgNode): config
            input_dir (str | Path): path to input dir
            output_dir (str | Path): path to output dir
            gen_page (GenPageXML): class to convert from predictions to pageXML
        """
        super().__init__(cfg)

        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        if not isinstance(output_page, OutputPageXML):
            raise TypeError(
                f"Must provide conversion from mask to pageXML. Current type is {type(output_page)}, not OutputPageXML"
            )

        self.output_page = output_page

        # Formats found here: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#imread
        self.image_formats = [
            ".bmp",
            ".dib",
            ".jpeg",
            ".jpg",
            ".jpe",
            ".jp2",
            ".png",
            ".webp",
            ".pbm",
            ".pgm",
            ".ppm",
            ".pxm",
            ".pnm",
            ".pfm",
            ".sr",
            ".ras",
            ".tiff",
            ".tif",
            ".exr",
            ".hdr",
            ".pic",
        ]
        # Formats https://pytorch.org/vision/main/generated/torchvision.io.read_image.html
        # self.image_formats = [".jpeg", ".jpg", ".jpe",
        #                       ".png"]

    def set_input_paths(
        self,
        input_paths: str | Path | Sequence[str | Path],
    ) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        input_paths = clean_input_paths(input_paths)

        all_input_paths = []

        for input_path in input_paths:
            if not input_path.exists():
                raise FileNotFoundError(f"Input ({input_path}) is not found")

            if not os.access(path=input_path, mode=os.R_OK):
                raise PermissionError(f"No access to {input_path} for read operations")

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
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()

    # def save_prediction(self, input_path: Path | str):
    def save_prediction(self, image, input_path):
        """
        Run the model and get the prediction, and save pageXML or a mask image depending on the mode

        Args:
            input_path (Path | str): path to single image

        Raises:
            TypeError: no output dir is specified
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        if image is None:
            self.logger.warning(f"Image at {input_path} has not loaded correctly, ignoring for now")
            return

        if isinstance(image, np.ndarray):
            outputs = self.cpu_call(image)
        elif isinstance(image, torch.Tensor):
            outputs = self.gpu_call(image)
        else:
            raise TypeError(f"Unknown image type: {type(image)}")

        output_image = outputs[0]["sem_seg"]
        # output_image = torch.argmax(output_image, dim=-3).cpu().numpy()

        self.output_page.link_image(input_path)
        self.output_page.generate_single_page(output_image, input_path, old_height=outputs[1], old_width=outputs[2])

    def process(self):
        """
        Run the model on all images within the input dir

        Raises:
            TypeError: no input dir is specified
            TypeError: no output dir is specified
        """
        if self.input_paths is None:
            raise TypeError("Cannot run when the input_paths is None")
        if self.output_dir is None:
            raise TypeError("Cannot run when the output_dir is None")

        input_paths = get_file_paths(self.input_paths, self.image_formats)

        # Single thread
        # for inputs in tqdm(input_paths):
        #     self.save_prediction(inputs)
        dataset = LoadingDataset(input_paths)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=None, num_workers=16, pin_memory=False, collate_fn=None)
        for inputs in tqdm(dataloader, desc="Predicting PageXML"):
            # self.logger.warning(inputs)
            self.save_prediction(inputs[0], inputs[1])

        # Multithread <- does not work with cuda
        # with Pool(os.cpu_count()) as pool:
        #     results = list(tqdm(pool.imap_unordered(
        #         self.save_prediction, image_paths), total=len(image_paths)))


def main(args: argparse.Namespace) -> None:
    cfg = setup_cfg(args)
    setup_logging(cfg, save_log=False)

    output_page = OutputPageXML(
        mode=cfg.MODEL.MODE,
        output_dir=args.output,
        line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
        regions=cfg.PREPROCESS.REGION.REGIONS,
        merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
        region_type=cfg.PREPROCESS.REGION.REGION_TYPE,
    )

    predictor = SavePredictor(
        cfg=cfg,
        input_paths=args.input,
        output_dir=args.output,
        output_page=output_page,
    )
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    predictor.process()

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    args = get_arguments()
    main(args)
