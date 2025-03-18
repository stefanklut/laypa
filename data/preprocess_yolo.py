import json
import logging
import os
import sys
from collections import Counter, defaultdict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict

import detectron2.data.transforms as T
import imagesize
import numpy as np
import torch
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as Pool

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from detectron2.config import CfgNode, configurable

from data.augmentations import Augmentation, build_augmentation
from data.mapper import AugInput
from page_xml.xml_converters import XMLToYOLO
from page_xml.xml_regions import XMLRegions
from utils.copy_utils import copy_mode
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name
from utils.path_utils import check_path_accessible, image_path_to_xml_path


class Image(TypedDict):
    """
    Required fields for an image dict
    """

    file_name: str
    height: int
    width: int
    id: Any


class PreprocessYOLO:
    """
    Used for almost all preprocessing steps to prepare datasets to be used by the training loop
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

        self.output = {"image": "png", "yolo": None}

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

        ret = {
            "augmentations": build_augmentation(cfg, "preprocess"),
            "input_paths": input_paths,
            "output_dir": output_dir,
            "xml_regions": XMLRegions(cfg),  # type: ignore
            "square_lines": cfg.PREPROCESS.BASELINE.SQUARE_LINES,
            "n_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "disable_check": cfg.PREPROCESS.DISABLE_CHECK,
            "overwrite": cfg.PREPROCESS.OVERWRITE,
            "output": {"image": "png", "yolo": None},
            "auto_dpi": cfg.PREPROCESS.DPI.AUTO_DETECT,
            "default_dpi": cfg.PREPROCESS.DPI.DEFAULT_DPI,
            "manual_dpi": cfg.PREPROCESS.DPI.MANUAL_DPI,
        }
        return ret

    def set_input_paths(
        self,
        input_paths: str | Path | Sequence[str | Path],
        ignore_duplicates: bool = False,
    ) -> None:
        """
        Setter of the input paths, turn string to path. And resolve full path

        Args:
            input_paths (str | Path | Sequence[str  |  Path]): path(s) from which to extract the images
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
        Getter of the input paths

        Returns:
            Optional[Sequence[Path]]: path(s) from which to extract the images
        """
        return self.input_paths

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter of output dir, turn string to path. And resolve full path

        Args:
            output_dir (str | Path): output path of the processed images
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
        Check if all paths given exist and are readable

        Args:
            paths (list[Path]): paths to be checked
        """
        all(check_path_accessible(path) for path in paths)

    def save_array_to_path(self, array: np.ndarray | torch.Tensor, path: Path) -> None:
        """
        Save an array to a path with a specific method

        Args:
            array (np.ndarray): array to be saved
            path (Path): path to save the array
        """

        method = path.suffix

        if method == ".png":
            if isinstance(array, torch.Tensor):
                array = array.permute(1, 2, 0).cpu().numpy()
            assert array.dtype == np.uint8, f"Array must be of type uint8 to save as PNG, got {array.dtype}"
            assert (
                array.ndim == 2 or array.shape[2] == 3
            ), f"Array must be 2D or 3D with 3 channels to save as PNG, got {array.shape}"
            assert np.max(array) <= 255, f"Array must be in range 0-255 to save as PNG, got {np.max(array)}"
            assert np.min(array) >= 0, f"Array must be in range 0-255 to save as PNG, got {np.min(array)}"
            assert path.suffix == ".png", f"Path must have suffix .png to save as PNG, got {path.suffix}"
            save_image_array_to_path(path, array)
        elif method == ".npy":
            if isinstance(array, torch.Tensor):
                array = array.permute(1, 2, 0).cpu().numpy()
            assert array.ndim == 3 or array.ndim == 2, f"Array must be 2D or 3D to save as numpy, got {array.shape}"
            assert path.suffix == ".npy", f"Path must have suffix .npy to save as numpy, got {path.suffix}"
            np.save(path, array)
        elif method == ".pt":
            if isinstance(array, torch.Tensor):
                tensor = array.cpu()
            else:
                tensor = torch.from_numpy(array)
                if array.ndim == 3:
                    tensor = tensor.permute(2, 0, 1)
            assert tensor.dim() == 3 or tensor.dim() == 2, f"Tensor must be 2D or 3D to save as torch, got {tensor.dim()}"
            assert path.suffix == ".pt", f"Path must have suffix .pt to save as torch, got {path.suffix}"
            torch.save(tensor, path)
        else:
            raise NotImplementedError(f"Method {method} not implemented")

    def save_image(
        self,
        image_path: Path,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
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

    def save_yolo(self, image_path: Path, original_image_shape: tuple[int, int], image_shape: tuple[int, int]):
        """
        Generate the YOLO format for an image.

        Args:
            image_path (Path): The path to the image file.
            original_image_shape (tuple[int, int]): The original shape of the image.
            image_shape (tuple[int, int]): The desired shape of the image.
        """
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        xml_path = image_path_to_xml_path(image_path, self.disable_check)

        yolo_dir = self.output_dir.joinpath("labels")

        out_yolo_path = yolo_dir.joinpath(xml_path.name).with_suffix(f".txt")
        out_yolo_size_path = yolo_dir.joinpath(xml_path.name).with_suffix(".size")

        # Check if image already exists and if it doesn't need resizing
        if not self.overwrite and out_yolo_path.exists() and out_yolo_size_path.exists():
            with out_yolo_size_path.open(mode="r") as f:
                out_yolo_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_yolo_shape == image_shape:
                return {"yolo_file_name": str(out_yolo_path.relative_to(self.output_dir))}

        converter = XMLToYOLO(self.xml_regions, square_lines=self.square_lines)

        yolo = converter.convert(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)

        if not yolo["annotations"]:
            return {"yolo_file_name": None}  # Skip empty annotations

        yolo_dir.mkdir(parents=True, exist_ok=True)
        with out_yolo_path.open(mode="w") as f:
            for annotation in yolo["annotations"]:
                output = [annotation["category_id"]] + annotation["bbox"]
                f.write(" ".join(map(str, output)) + "\n")

        return {"yolo_file_name": str(out_yolo_path.relative_to(self.output_dir))}

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

    def run(self):
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
