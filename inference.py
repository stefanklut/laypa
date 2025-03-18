import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Type, Union

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from tqdm import tqdm

from core.setup import setup_cfg, setup_logging
from data.augmentations import build_augmentation
from data.mapper import AugInput
from page_xml.output_pageXML import OutputPageXML
from page_xml.xml_regions import XMLRegions
from utils.image_utils import load_image_array_from_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file to inference using the model found in the config file")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", help="optional args to change", action="extend", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Input folder",
        type=str,
        action="extend",
        required=True,
    )
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    page_xml_args = parser.add_argument_group("PageXML")
    page_xml_args.add_argument("-w", "--whitelist", nargs="+", help="Input folder", type=str, action="extend")
    page_xml_args.add_argument(
        "--save_confidence_heatmap",
        help="Save the confidence heatmap",
        action="store_true",
    )

    dataloader_args = parser.add_argument_group("Dataloader")
    dataloader_args.add_argument("--num_workers", help="Number of workers to use", type=int, default=4)

    args = parser.parse_args()

    return args


class Predictor(DefaultPredictor):
    """
    Predictor runs the model specified in the config, on call the image is processed and the results dict is output
    """

    def __init__(self, cfg: CfgNode):
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

        precision_converter = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.precision = precision_converter.get(cfg.MODEL.AMP_TEST.PRECISION, None)
        if self.precision is None:
            raise ValueError(f"Unrecognized precision: {cfg.MODEL.AMP_TEST.PRECISION}")

        assert self.cfg.INPUT.FORMAT in ["RGB", "BGR"], self.cfg.INPUT.FORMAT

        checkpointer = DetectionCheckpointer(self.model)
        if not cfg.TEST.WEIGHTS:
            raise FileNotFoundError("Cannot do inference without weights. Specify a checkpoint file to --opts TEST.WEIGHTS")

        checkpointer.load(cfg.TEST.WEIGHTS)

        self.aug = T.AugmentationList(build_augmentation(cfg, "test"))

    # def gpu_call(self, original_image: torch.Tensor) -> tuple[dict, int, int]:
    #     """
    #     Run the model on the image with preprocessing on the gpu

    #     Args:
    #         original_image (torch.Tensor): image to run the model on

    #     Returns:
    #         tuple[dict, int, int]: predictions, height, width
    #     """
    #     with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
    #         # Apply pre-processing to image.
    #         channels, height, width = original_image.shape
    #         assert channels == 3, f"Must be a BGR image, found {channels} channels"
    #         image = torch.as_tensor(original_image, dtype=torch.float32, device=self.cfg.MODEL.DEVICE)

    #         if self.cfg.INPUT.FORMAT == "BGR":
    #             # whether the model expects BGR inputs or RGB
    #             image = image[[2, 1, 0], :, :]

    #         new_height, new_width = self.get_image_size(height, width)

    #         if self.cfg.INPUT.RESIZE_MODE != "none":
    #             image = torch.nn.functional.interpolate(image[None], mode="bilinear", size=(new_height, new_width))[0]

    #         inputs = {"image": image, "height": new_height, "width": new_width}

    #         with torch.autocast(
    #             device_type=self.cfg.MODEL.DEVICE,
    #             enabled=self.cfg.MODEL.AMP_TEST.ENABLED,
    #             dtype=self.precision,
    #         ):
    #             predictions = self.model([inputs])[0]

    #         # if torch.isnan(predictions["sem_seg"]).any():
    #         #     raise ValueError("NaN in predictions")

    #         return predictions, height, width

    def cpu_call(self, data: AugInput, device: Optional[str] = None) -> tuple[dict, int, int]:
        """
        Run the model on the image with preprocessing on the cpu

        Args:
            data (AugInput): image to run the model on
            device (str): device to run the model on

        Returns:
            tuple[dict, int, int]: predictions, height, width
        """
        logger = logging.getLogger(get_logger_name())

        # Default value of device should be the one in the config
        if device is None:
            device = self.cfg.MODEL.DEVICE

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.

            height, width, channels = data.image.shape
            assert channels == 3, f"Must be a RBG image, found {channels} channels"
            # In place augmentation
            transform = self.aug(data)
            image = torch.as_tensor(data.image, dtype=torch.float32, device=device).permute(2, 0, 1)

            if self.cfg.INPUT.FORMAT == "BGR":
                # whether the model expects BGR inputs or RGB
                image = image[[2, 1, 0], :, :]

            inputs = {"image": image, "height": image.shape[1], "width": image.shape[2]}

            # If we predict on CPU, use full precision
            precision = self.precision if device != "cpu" else torch.float32

            with torch.autocast(
                device_type=device,
                enabled=self.cfg.MODEL.AMP_TEST.ENABLED,
                dtype=precision,
            ):
                self.model.to(device)

                predictions = self.model([inputs])[0]

            # if torch.isnan(predictions["sem_seg"]).any():
            #     raise ValueError("NaN in predictions")

        return predictions, height, width

    def __call__(self, data: AugInput, device: Optional[str] = None) -> tuple[dict, int, int]:
        """
        Run the model on the image with preprocessing

        Args:
            data (AugInput): image (and possibly dpi) to run the model on
            device (str): device to run the model on
        Returns:
            tuple[dict, int, int]: predictions, height, width
        """

        # if isinstance(original_image, np.ndarray):
        #     return self.cpu_call(original_image)
        # elif isinstance(original_image, torch.Tensor):
        #     return self.gpu_call(original_image)
        # else:
        #     raise TypeError(f"Unknown image type: {type(original_image)}")
        return self.cpu_call(data, device)


class LoadingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]

        # TODO Move resize and load to this part of the dataloader
        data = load_image_array_from_path(path)
        if data is None:
            return None, None, path
        image = data["image"]
        dpi = data["dpi"]
        return image, dpi, path


def collate_numpy(batch):
    collate_map = default_collate_fn_map

    def new_map(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
    ):
        return batch

    collate_map.update({np.ndarray: new_map, type(None): new_map})
    return collate(batch, collate_fn_map=collate_map)


class SavePredictor(Predictor):
    """
    Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML
    """

    def __init__(
        self,
        cfg: CfgNode,
        input_paths: str | Path | Sequence[str | Path],
        output_dir: str | Path,
        output_page: OutputPageXML,
        num_workers: int = 4,
    ):
        """
        Extension on the predictor that actually saves the part on the prediction we current care about: the semantic segmentation as pageXML

        Args:
            cfg (CfgNode): config
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images
            output_dir (str | Path): path to output dir
            output_page (OutputPageXML): output pageXML object
            num_workers (int): number of workers to use

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

        self.num_workers = num_workers

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
        self.input_paths = get_file_paths(input_paths, SUPPORTED_IMAGE_FORMATS)

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
    def save_prediction(self, image: np.ndarray, dpi: int, input_path: Path):
        """
        Run the model on the image and save the results as pageXML

        Args:
            image (np.ndarray): image to run the model on
            dpi (int): dpi of the image
            input_path (Path): path to the image

        Raises:
            TypeError: no input dir is specified
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        if image is None:
            self.logger.warning(f"Image at {input_path} has not loaded correctly, ignoring for now")
            return

        data = AugInput(
            image,
            dpi=dpi,
            auto_dpi=self.cfg.INPUT.DPI.AUTO_DETECT_TEST,
            default_dpi=self.cfg.INPUT.DPI.DEFAULT_DPI_TEST,
            manual_dpi=self.cfg.INPUT.DPI.MANUAL_DPI_TEST,
        )

        outputs = self.__call__(data)

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

        dataset = LoadingDataset(self.input_paths)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=collate_numpy,
        )
        for inputs in tqdm(dataloader, desc="Predicting PageXML"):
            self.save_prediction(inputs[0], inputs[1], inputs[2])


def main(args: argparse.Namespace) -> None:
    cfg = setup_cfg(args)
    setup_logging(cfg, save_log=False)
    xml_regions = XMLRegions(cfg)  # type: ignore
    output_page = OutputPageXML(
        xml_regions=xml_regions,
        output_dir=args.output,
        cfg=cfg,
        whitelist=args.whitelist,
        rectangle_regions=cfg.PREPROCESS.REGION.RECTANGLE_REGIONS,
        min_region_size=cfg.PREPROCESS.REGION.MIN_REGION_SIZE,
        save_confidence_heatmap=args.save_confidence_heatmap,
    )

    predictor = SavePredictor(
        cfg=cfg,
        input_paths=args.input,
        output_dir=args.output,
        output_page=output_page,
        num_workers=args.num_workers,
    )
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, record_shapes=True) as prof:
    predictor.process()

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))


if __name__ == "__main__":
    args = get_arguments()
    main(args)
