import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import detectron2.data.transforms as T
import imagesize
import numpy as np
import torch
from tqdm import tqdm

from page_xml.xml_converters import (
    XMLToBinarySeg,
    XMLToInstances,
    XMLToPano,
    XMLToSemSeg,
)

# from multiprocessing.pool import ThreadPool as Pool

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from detectron2.config import CfgNode, configurable

from data.augmentations import (
    Augmentation,
    ResizeLongestEdge,
    ResizeScaling,
    ResizeShortestEdge,
    build_augmentation,
)
from data.mapper import AugInput
from page_xml.xml_converters.xml_converter import _XMLConverter
from page_xml.xml_regions import XMLRegions
from utils.copy_utils import copy_mode
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name
from utils.path_utils import check_path_accessible, image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        parents=[Preprocess.get_parser(), XMLRegions.get_parser()],
        description="Preprocessing an annotated dataset of documents with pageXML",
    )

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)
    io_args.add_argument("-o", "--output", help="Output folder", required=True, type=str)

    xml_converter_args = parser.add_argument_group("XML Converter")
    xml_converter_args.add_argument("--square-lines", help="Square the lines", action="store_true")

    args = parser.parse_args()
    return args


class Preprocess:
    """
    Used for almost all preprocessing steps to prepare datasets to be used by the training loop.
    """

    @configurable
    def __init__(
        self,
        augmentations: list[Augmentation],
        input_paths: Optional[Sequence[Path]] = None,
        output_dir: Optional[Path] = None,
        xml_regions: Optional[XMLRegions] = None,
        square_lines: bool = False,
        n_classes: Optional[int] = None,
        disable_check: bool = False,
        overwrite: bool = False,
        output: dict[str, str] = {},
        auto_dpi: bool = True,
        default_dpi: Optional[int] = None,
        manual_dpi: Optional[int] = None,
    ) -> None:
        """
        Initializes the Preprocessor object.

        Args:
            augmentations (list[Augmentation]): List of augmentations to be applied during preprocessing.
            input_paths (Sequence[Path], optional): The input directory or files used to generate the dataset. Defaults to None.
            output_dir (Path, optional): The destination directory of the generated dataset. Defaults to None.
            xml_converter (XMLConverter, optional): The converter used to convert XML to image. Defaults to None.
            n_classes (int, optional): The number of classes in the dataset. Defaults to None.
            disable_check (bool, optional): Flag to turn off filesystem checks. Defaults to False.
            overwrite (bool, optional): Flag to force overwrite of images. Defaults to False.
            auto_dpi (bool, optional): Flag to automatically determine the DPI of the images. Defaults to True.
            default_dpi (int, optional): The default DPI to be used for resizing images. Defaults to None.
            manual_dpi (int, optional): The manually specified DPI to be used for resizing images. Defaults to None.

        Raises:
            TypeError: If xml_converter is not an instance of XMLConverter.
            AssertionError: If the number of specified regions does not match the number of specified classes.

        """
        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        self.disable_check = disable_check
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        assert isinstance(xml_regions, XMLRegions), f"xml_regions must be an instance of XMLRegions, got {type(xml_regions)}"

        self.xml_regions = xml_regions

        self.square_lines = square_lines

        if n_classes is not None:
            assert (n_regions := len(xml_regions.regions)) == (
                n_classes
            ), f"Number of specified regions ({n_regions}) does not match the number of specified classes ({n_classes})"

        self.overwrite = overwrite

        self.output = output

        self.augmentations = augmentations

        self.auto_dpi = auto_dpi
        self.default_dpi = default_dpi
        self.manual_dpi = manual_dpi

    @classmethod
    def from_config(
        cls,
        cfg: CfgNode,
        input_paths: Optional[Sequence[Path]] = None,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """
        Converts a configuration object to a dictionary to be used as keyword arguments.

        Args:
            cfg (CfgNode): The configuration object.
            input_paths (Optional[Sequence[Path]], optional): The input directory or files used to generate the dataset. Defaults to None.
            output_dir (Optional[Path], optional): The destination directory of the generated dataset. Defaults to None.

        Returns:
            dict[str, Any]: A dictionary containing the converted configuration values.
        """

        n_classes = None
        if cfg.MODEL.META_ARCHITECTURE == "SemanticSegmentor":
            n_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        elif cfg.MODEL.META_ARCHITECTURE == "BinarySegmentor":
            n_classes = cfg.MODEL.BINARY_SEG_HEAD.NUM_CLASSES + 1
        else:
            raise NotImplementedError(f"Meta architecture {cfg.MODEL.META_ARCHITECTURE} not implemented")

        ret = {
            "augmentations": build_augmentation(cfg, "preprocess"),
            "input_paths": input_paths,
            "output_dir": output_dir,
            "xml_regions": XMLRegions(cfg),
            "square_lines": cfg.PREPROCESS.BASELINE.SQUARE_LINES,
            "n_classes": n_classes,
            "disable_check": cfg.PREPROCESS.DISABLE_CHECK,
            "overwrite": cfg.PREPROCESS.OVERWRITE,
            "output": {key: value for key, value in cfg.PREPROCESS.OUTPUT},
            "auto_dpi": cfg.PREPROCESS.DPI.AUTO_DETECT,
            "default_dpi": cfg.PREPROCESS.DPI.DEFAULT_DPI,
            "manual_dpi": cfg.PREPROCESS.DPI.MANUAL_DPI,
        }
        return ret

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """
        Return argparser that has the arguments required for the preprocessing.

        Returns:
            argparse.ArgumentParser: the argparser for preprocessing.
        """
        parser = argparse.ArgumentParser(add_help=False)
        pre_process_args = parser.add_argument_group("preprocessing")

        pre_process_args.add_argument("--resize", action="store_true", help="Resize input images")
        pre_process_args.add_argument(
            "--resize_mode",
            default="none",
            choices=["none", "shortest_edge", "longest_edge", "scaling"],
            type=str,
            help="How to select the size when resizing",
        )
        pre_process_args.add_argument(
            "--resize_sampling",
            default="choice",
            choices=["range", "choice"],
            type=str,
            help="How to select the size when resizing",
        )
        pre_process_args.add_argument(
            "--scaling", default=0.5, type=float, help="Scaling factor while resizing with mode scaling"
        )

        pre_process_args.add_argument("--min_size", default=[1024], nargs="*", type=int, help="Min resize shape")
        pre_process_args.add_argument("--max_size", default=2048, type=int, help="Max resize shape")

        pre_process_args.add_argument("--disable_check", action="store_true", help="Don't check if all images exist")

        pre_process_args.add_argument("--overwrite", action="store_true", help="Overwrite the images and label masks")

        pre_process_args.add_argument("--auto_dpi", action="store_true", help="Automatically detect DPI")
        pre_process_args.add_argument("--default_dpi", type=int, help="Default DPI")
        pre_process_args.add_argument("--manual_dpi", type=int, help="Manually set DPI")

        return parser

    def set_input_paths(
        self,
        input_paths: str | Path | Sequence[str | Path],
        ignore_duplicates: bool = False,
    ) -> None:
        """
        Setter of the input paths, turn string to path. And resolve full path.

        Args:
            input_paths (str | Path | Sequence[str  |  Path]): path(s) from which to extract the images.
            ignore_duplicates (bool, optional): Ignore duplicate names in the input paths. Defaults to False.
        """
        input_paths = get_file_paths(input_paths, SUPPORTED_IMAGE_FORMATS, self.disable_check)
        if not ignore_duplicates:
            self.check_duplicates(input_paths)

        self.input_paths = input_paths

    def check_duplicates(
        self,
        input_paths: Sequence[Path],
    ) -> None:
        """
        Check for duplicate names in a list of input paths.

        Args:
            input_paths (Sequence[Path]): A sequence of Path objects representing the input paths.

        Raises:
            ValueError: If duplicate names are found in the input paths.
        """
        count_duplicates_names = Counter([path.name for path in input_paths])
        duplicates = defaultdict(list)
        for path in input_paths:
            if count_duplicates_names[path.name] > 1:
                duplicates[path.name].append(path)
        if duplicates:
            total_duplicates = sum(count_duplicates_names[name] for name in duplicates.keys())
            count_per_dir = Counter([path.parent for path in input_paths])
            duplicates_in_dir = defaultdict(int)
            duplicates_makeup = defaultdict(lambda: defaultdict(int))
            for name, paths in duplicates.items():
                for path in paths:
                    duplicates_in_dir[path.parent] += 1
                    for other_path in duplicates[name]:
                        if other_path.parent != path.parent:
                            duplicates_makeup[path.parent][other_path.parent] += 1
            duplicate_warning = "Duplicates found in the following directories:\n"
            for dir_path, count in count_per_dir.items():
                duplicate_warning += f"Directory: {dir_path} Count: {duplicates_in_dir.get(dir_path, 0)}/{count}\n"
                if dir_path in duplicates_makeup:
                    duplicate_warning += "Shared Duplicates:"
                    for other_dir, makeup_count in duplicates_makeup[dir_path].items():
                        duplicate_warning += f"\n\t{other_dir} Count: {makeup_count}/{count}"
                    duplicate_warning += "\n"
            self.logger.warning(duplicate_warning.strip())
            raise ValueError(
                f"Found duplicate names in input paths. \n\tDuplicates: {total_duplicates}/{len(input_paths)} \n\tTotal unique names: {len(count_duplicates_names)}"
            )

    def get_input_paths(self) -> Optional[Sequence[Path]]:
        """
        Getter of the input paths.

        Returns:
            Optional[Sequence[Path]]: path(s) from which to extract the images.
        """
        return self.input_paths

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter of output dir, turn string to path. And resolve full path.

        Args:
            output_dir (str | Path): output path of the processed images.
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()

    def get_output_dir(self) -> Optional[Path]:
        """
        Getter of the output dir

        Returns:
            Optional[Path]: output path of the processed images
        """
        return self.output_dir

    @staticmethod
    def check_paths_exists(paths: Sequence[Path]) -> None:
        """
        Check if all paths given exist and are readable.
        Args:
            paths (list[Path]): paths to be checked.
        """
        all(check_path_accessible(path) for path in paths)

    def save_array_to_path(self, array: np.ndarray | torch.Tensor, path: Path) -> None:
        """
        Save an array to a path with a specific method.

        Args:
            array (np.ndarray): array to be saved.
            path (Path): path to save the array.
        """

        method = path.suffix

        if method in [".png", ".jpg", ".jpeg", ".webp"]:
            if isinstance(array, torch.Tensor):
                array = array.permute(1, 2, 0).cpu().numpy()
            assert array.dtype == np.uint8, f"Array must be of type uint8 to save an image, got {array.dtype}"
            assert (
                array.ndim == 2 or array.shape[2] == 3
            ), f"Array must be 2D or 3D with 3 channels to save an image, got {array.shape}"
            assert np.max(array) <= 255, f"Array must be in range 0-255 to save an image, got {np.max(array)}"
            assert np.min(array) >= 0, f"Array must be in range 0-255 to save an image, got {np.min(array)}"
            save_image_array_to_path(path, array)
        elif method == ".npy":
            if isinstance(array, torch.Tensor):
                array = array.permute(1, 2, 0).cpu().numpy()
            assert array.ndim == 3 or array.ndim == 2, f"Array must be 2D or 3D to save as numpy, got {array.shape}"
            np.save(path, array)
        elif method == ".pt":
            if isinstance(array, torch.Tensor):
                tensor = array.cpu()
            else:
                tensor = torch.from_numpy(array)
                if array.ndim == 3:
                    tensor = tensor.permute(2, 0, 1)
            assert tensor.dim() == 3 or tensor.dim() == 2, f"Tensor must be 2D or 3D to save as torch, got {tensor.dim()}"
            torch.save(tensor, path)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def save_image(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Optional[dict[str, str]]:
        """
        Save an image to the output directory.

        Args:
            image_path (Path): The path to the original image file.
            image_stem (str): The stem of the image file name.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.

        Returns:
            dict: The relative path to the saved image.

        Raises:
            TypeError: If the output directory is None.
            TypeError: If the image loading fails.
        """

        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        image_dir = self.output_dir.joinpath("image")
        save_method_image = self.output["image"]

        copy_image = True if original_image_shape == image_shape else False

        if copy_image:
            out_image_path = image_dir.joinpath(image_path.name)
        else:
            out_image_path = image_dir.joinpath(image_path.name).with_suffix(f".{save_method_image}")

        out_image_size_path = image_dir.joinpath(image_path.name).with_suffix(".size")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_image_path.exists() and out_image_size_path.exists():
            with out_image_size_path.open(mode="r") as f:
                out_image_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_image_shape == image_shape:
                return {"image_file_name": str(out_image_path.relative_to(self.output_dir))}

        image_dir.mkdir(parents=True, exist_ok=True)

        if copy_image:
            copy_mode(image_path, out_image_path, mode="link")
        else:
            data = load_image_array_from_path(image_path)
            if data is None:
                raise TypeError(f"Image {image_path} is None, loading failed")
            aug_input = AugInput(
                data["image"],
                dpi=data["dpi"],
                auto_dpi=self.auto_dpi,
                default_dpi=self.default_dpi,
                manual_dpi=self.manual_dpi,
            )
            transforms = T.AugmentationList(self.augmentations)(aug_input)
            self.save_array_to_path(aug_input.image, out_image_path)

        with out_image_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        results = {"image_file_name": str(out_image_path.relative_to(self.output_dir))}

        return results

    def save_sem_seg(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Optional[dict[str, str]]:
        """
        Save the semantic segmentation mask for an image.

        Args:
            image_path (Path): The path to the image file.
            image_stem (str): The stem of the image file name.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.

        Returns:
            dict: The relative path to the saved semantic segmentation mask.

        Raises:
            TypeError: If the output directory is None.

        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        xml_path = image_path_to_xml_path(image_path, self.disable_check)

        sem_seg_dir = self.output_dir.joinpath("sem_seg")
        save_method_sem_seg = self.output["sem_seg"]

        out_sem_seg_path = sem_seg_dir.joinpath(xml_path.name).with_suffix(f".{save_method_sem_seg}")
        out_sem_seg_size_path = sem_seg_dir.joinpath(xml_path.name).with_suffix(".size")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_sem_seg_path.exists() and out_sem_seg_size_path.exists():
            with out_sem_seg_size_path.open(mode="r") as f:
                out_sem_seg_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_sem_seg_shape == image_shape:
                return {"sem_seg_file_name": str(out_sem_seg_path.relative_to(self.output_dir))}

        converter = XMLToSemSeg(self.xml_regions, square_lines=self.square_lines)

        sem_seg = converter.convert(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if sem_seg is None:
            return None

        sem_seg_dir.mkdir(parents=True, exist_ok=True)

        self.save_array_to_path(sem_seg, out_sem_seg_path)

        with out_sem_seg_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        results = {"sem_seg_file_name": str(out_sem_seg_path.relative_to(self.output_dir))}

        return results

    def save_instances(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Optional[dict[str, str]]:
        """
        Save instances to JSON file.

        Args:
            image_path (Path): The path to the image file.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.

        Returns:
            dict: The relative path to the saved instances file.

        Raises:
            ValueError: If the output directory is not set.

        """
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")

        xml_path = image_path_to_xml_path(image_path, self.disable_check)

        instances_dir = self.output_dir.joinpath("instances")
        save_method_instances = self.output["instances"]

        out_instances_path = instances_dir.joinpath(xml_path.name).with_suffix(f".{save_method_instances}")
        out_instances_size_path = instances_dir.joinpath(xml_path.name).with_suffix(".size")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_instances_path.exists() and out_instances_size_path.exists():
            with out_instances_size_path.open(mode="r") as f:
                out_intstances_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_intstances_shape == image_shape:
                return {"annotations": str(out_instances_path.relative_to(self.output_dir))}

        converter = XMLToInstances(self.xml_regions, square_lines=self.square_lines)

        instances = converter.convert(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if instances is None:
            return None

        instances_dir.mkdir(parents=True, exist_ok=True)

        with out_instances_path.open(mode="w") as f:
            json.dump(instances, f)
        with out_instances_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        results = {"annotations": str(out_instances_path.relative_to(self.output_dir))}

        return results

    def save_pano(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Optional[dict[str, str]]:
        """
        Save panoramic image and segments information to the output directory.

        Args:
            image_path (Path): The path to the image file.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.

        Returns:
            dict: The relative paths to the saved panoramic image and segments information.

        Raises:
            TypeError: If the output directory is None.

        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        xml_path = image_path_to_xml_path(image_path, self.disable_check)

        pano_dir = self.output_dir.joinpath("pano")
        save_method_pano = self.output["pano"]

        out_pano_path = pano_dir.joinpath(xml_path.name).with_suffix(f".{save_method_pano}")
        out_pano_size_path = pano_dir.joinpath(xml_path.name).with_suffix(".size")
        out_segments_info_path = pano_dir.joinpath(xml_path.name).with_suffix(".json")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_pano_path.exists() and out_segments_info_path.exists() and out_pano_size_path.exists():
            with out_pano_size_path.open(mode="r") as f:
                out_pano_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_pano_shape == image_shape:
                return {
                    "pano_file_name": str(out_pano_path.relative_to(self.output_dir)),
                    "segments_info": str(out_segments_info_path.relative_to(self.output_dir)),
                }

        converter = XMLToPano(self.xml_regions, square_lines=self.square_lines)

        pano_output = converter.convert(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if pano_output is None:
            return None
        pano, segments_info = pano_output

        pano_dir.mkdir(parents=True, exist_ok=True)

        self.save_array_to_path(pano, out_pano_path)

        with out_pano_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        with out_segments_info_path.open(mode="w") as f:
            json.dump(segments_info, f)

        results = {
            "pano_file_name": str(out_pano_path.relative_to(self.output_dir)),
            "segments_info": str(out_segments_info_path.relative_to(self.output_dir)),
        }

        return results

    def save_binary_seg(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ) -> Optional[dict[str, str]]:
        """
        Save the binary segmentation mask for an image.

        Args:
            image_path (Path): The path to the image file.
            image_stem (str): The stem of the image file name.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.

        Returns:
            dict: The relative path to the saved binary segmentation mask.

        Raises:
            TypeError: If the output directory is None.

        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        xml_path = image_path_to_xml_path(image_path, self.disable_check)

        binary_seg_dir = self.output_dir.joinpath("binary_seg")
        save_method_binary_seg = self.output["binary_seg"]

        out_binary_seg_path = binary_seg_dir.joinpath(xml_path.name).with_suffix(f".{save_method_binary_seg}")
        out_binary_seg_size_path = binary_seg_dir.joinpath(xml_path.name).with_suffix(".size")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_binary_seg_path.exists() and out_binary_seg_size_path.exists():
            with out_binary_seg_size_path.open(mode="r") as f:
                out_binary_seg_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_binary_seg_shape == image_shape:
                return {"binary_seg_file_name": str(out_binary_seg_path.relative_to(self.output_dir))}
        converter = XMLToBinarySeg(self.xml_regions, square_lines=self.square_lines)

        binary_seg = converter.convert(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if binary_seg is None:
            return None

        binary_seg_dir.mkdir(parents=True, exist_ok=True)

        self.save_array_to_path(binary_seg, out_binary_seg_path)

        with out_binary_seg_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        results = {"binary_seg_file_name": str(out_binary_seg_path.relative_to(self.output_dir))}

        return results

    def get_dpi(self, image_path: Path) -> Optional[int]:
        """
        Get the DPI of an image.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            int: The DPI of the image.

        """
        if self.auto_dpi:
            original_image_dpi = imagesize.getDPI(image_path)
            if original_image_dpi == (-1, -1):
                return self.default_dpi

            assert len(original_image_dpi) == 2, f"Invalid DPI: {original_image_dpi}"
            assert original_image_dpi[0] == original_image_dpi[1], f"Non-square DPI: {original_image_dpi}"
            original_image_dpi = original_image_dpi[0]
        else:
            original_image_dpi = self.manual_dpi

    def process_single_file(self, image_path: Path) -> dict:
        """
        Process a single image and pageXML to be used during training

        Args:
            image_path (Path): Path to input image

        Raises:
            TypeError: Cannot return if output dir is not set

        Returns:
            dict: Preprocessing results
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        image_stem = image_path.stem

        _original_image_shape = imagesize.get(image_path)
        original_image_shape = int(_original_image_shape[1]), int(_original_image_shape[0])
        original_image_dpi = self.get_dpi(image_path)
        image_shape = self.augmentations[0].get_output_shape(
            original_image_shape[0], original_image_shape[1], dpi=original_image_dpi
        )

        results = {}
        results["original_file_name"] = str(image_path)

        for output in self.output:
            if hasattr(self, f"save_{output}"):
                output_result = getattr(self, f"save_{output}")(image_path, original_image_shape, image_shape)
                if output_result is not None:
                    results.update(output_result)
            else:
                raise NotImplementedError(f"Output {output} not implemented")

        return results

    def run(self) -> None:
        """
        Run preprocessing on all images currently on input paths, save to output dir

        Raises:
            TypeError: Input paths must be set
            TypeError: Output dir must be set
            ValueError: Must find at least one image in all input paths
            ValueError: Must find at least one pageXML in all input paths
        """
        if self.input_paths is None:
            raise TypeError("Cannot run when the input path is None")
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        xml_paths = [image_path_to_xml_path(image_path, self.disable_check) for image_path in self.input_paths]

        if len(self.input_paths) == 0:
            raise ValueError(f"No images found when checking input ({self.input_paths})")

        if len(xml_paths) == 0:
            raise ValueError(f"No pagexml found when checking input  ({self.input_paths})")

        if not self.disable_check:
            self.check_paths_exists(self.input_paths)
            self.check_paths_exists(xml_paths)

        mode_path = self.output_dir.joinpath("mode.txt")

        if mode_path.exists():
            with mode_path.open(mode="r") as f:
                mode = f.read()
            if mode != self.xml_regions.mode:
                self.overwrite = True

        with mode_path.open(mode="w") as f:
            f.write(self.xml_regions.mode)

        # Single thread
        # results = []
        # for image_path in tqdm(image_paths, desc="Preprocessing"):
        #     results.append(self.process_single_file(image_path))

        # Multithread
        with Pool(os.cpu_count()) as pool:
            results = list(
                tqdm(
                    iterable=pool.imap_unordered(self.process_single_file, self.input_paths),
                    total=len(self.input_paths),
                    desc="Preprocessing",
                )
            )

        # Assuming all key are the same make one dict
        results = {
            "data": list_of_dict_to_dict_of_list(results),
            "classes": self.xml_regions.regions,
            "mode": self.xml_regions.mode,
        }

        output_path = self.output_dir.joinpath("info.json")
        with output_path.open(mode="w") as f:
            json.dump(results, f)


def list_of_dict_to_dict_of_list(input_list: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert a list of dicts into dict of lists. All dicts much have the same keys. The output number of dicts matches the length of the list

    Args:
        input_list (list[dict[str, Any]]): list of dicts

    Returns:
        dict[str, list[Any]]: dict of lists
    """
    output_dict = {key: [item[key] for item in input_list] for key in input_list[0].keys()}
    return output_dict


def main(args) -> None:
    xml_regions = XMLRegions(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type,
    )

    resize_mode = args.resize_mode
    resize_sampling = args.resize_sampling
    min_size = args.min_size
    max_size = args.max_size
    scaling = args.scaling

    augmentations = []
    if resize_mode == "none":
        augmentations.append(ResizeScaling(scale=1))
    elif resize_mode == "shortest_edge":
        augmentations.append(ResizeShortestEdge(min_size=min_size, max_size=max_size, sample_style=resize_sampling))
    elif resize_mode == "longest_edge":
        augmentations.append(ResizeLongestEdge(min_size=min_size, max_size=max_size, sample_style=resize_sampling))
    elif resize_mode == "scaling":
        augmentations.append(ResizeScaling(scale=scaling, max_size=max_size))
    else:
        raise NotImplementedError(f"Resize mode {resize_mode} not implemented")

    process = Preprocess(
        augmentations=augmentations,
        input_paths=args.input,
        output_dir=args.output,
        xml_regions=xml_regions,
        square_lines=args.square_lines,
        disable_check=args.disable_check,
        overwrite=args.overwrite,
        auto_dpi=args.auto_dpi,
        default_dpi=args.default_dpi,
        manual_dpi=args.manual_dpi,
    )
    process.run()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
