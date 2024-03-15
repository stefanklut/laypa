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
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as Pool

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from datasets.augmentations import (
    Augmentation,
    ResizeLongestEdge,
    ResizeScaling,
    ResizeShortestEdge,
)
from datasets.mapper import AugInput
from datasets.transforms import ResizeTransform
from page_xml.xml_converter import XMLConverter
from page_xml.xml_regions import XMLRegions
from utils.copy_utils import copy_mode
from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import get_file_paths, supported_image_formats
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
    Used for almost all preprocessing steps to prepare datasets to be used by the training loop
    """

    def __init__(
        self,
        augmentations: list[Augmentation],
        input_paths=None,
        output_dir=None,
        xml_converter=None,
        disable_check=False,
        overwrite=False,
    ) -> None:
        """
        Used for almost all preprocessing steps to prepare datasets to be used by the training loop

        Args:
            input_paths (Sequence[Path], optional): the used input dir/files to generate the dataset. Defaults to None.
            output_dir (Path, optional): the destination dir of the generated dataset. Defaults to None.
            resize_mode (str, optional): resize images before saving. Defaults to none.
            resize_sampling (str, optional): sample type used when resizing. Defaults to "choice".
            min_size (list, optional): when resizing, the length the shortest edge is resized to. Defaults to [1024].
            max_size (int, optional): when resizing, the max length a side may have. Defaults to 2048.
            xml_converter (XMLConverter, optional): converter to convert xml to image. Defaults to None.
            disable_check (bool, optional): flag to turn of filesystem checks, useful if run was already successful once. Defaults to False.
            overwrite (bool, optional): flag to force overwrite of images. Defaults to False.

        Raises:
            TypeError: Did not provide a XMLImage object to convert from XML to image
            ValueError: If resize mode is choice must provide the min size with 2 or more values
            ValueError: If resize mode is range must provide the min size with 2 values
            NotImplementedError: resize mode given is not
        """

        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        self.disable_check = disable_check
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        if not isinstance(xml_converter, XMLConverter):
            raise TypeError(f"Must provide conversion from xml to image. Current type is {type(xml_converter)}, not XMLImage")

        self.xml_converter = xml_converter

        self.overwrite = overwrite

        self.augmentations = augmentations

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """
        Return argparser that has the arguments required for the preprocessing.

        Returns:
            argparse.ArgumentParser: the argparser for preprocessing
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

        return parser

    def set_input_paths(self, input_paths: str | Path | Sequence[str | Path], ignore_duplicates=False) -> None:
        """
        Setter of the input paths, turn string to path. And resolve full path

        Args:
            input_paths (str | Path | Sequence[str  |  Path]): path(s) from which to extract the images
        """
        input_paths = get_file_paths(input_paths, supported_image_formats, self.disable_check)
        if not ignore_duplicates:
            self.check_duplicates(input_paths)

        self.input_paths = input_paths

    def check_duplicates(self, input_paths: Sequence[Path]) -> None:
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

    def save_image(
        self,
        image_path: Path,
        image_stem: str,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")

        image_dir = self.output_dir.joinpath("original")

        copy_image = True if original_image_shape == image_shape else False

        if copy_image:
            out_image_path = image_dir.joinpath(image_path.name)
        else:
            out_image_path = image_dir.joinpath(image_stem + ".png")

        # Check if image already exist and if it doesn't need resizing
        if not self.overwrite and out_image_path.exists():
            out_image_shape = imagesize.get(out_image_path)[::-1]
            if out_image_shape == image_shape:
                return str(out_image_path.relative_to(self.output_dir))

        image_dir.mkdir(parents=True, exist_ok=True)

        if copy_image:
            copy_mode(image_path, out_image_path, mode="link")
        else:
            data = load_image_array_from_path(image_path)
            if data is None:
                raise TypeError(f"Image {image_path} is None, loading failed")
            aug_input = AugInput(data["image"], sem_seg=None, dpi=data["dpi"])
            image = T.AugmentationList(self.augmentations)(aug_input).image
            save_image_array_to_path(out_image_path, image)

        return str(out_image_path.relative_to(self.output_dir))

    def save_sem_seg(
        self, xml_path: Path, image_stem: str, original_image_shape: tuple[int, int], image_shape: tuple[int, int]
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        sem_seg_dir = self.output_dir.joinpath("sem_seg")
        out_sem_seg_path = sem_seg_dir.joinpath(image_stem + ".png")

        # Check if image already exist and if it doesn't need resizing
        if not self.overwrite and out_sem_seg_path.exists():
            out_sem_seg_shape = imagesize.get(out_sem_seg_path)[::-1]
            if out_sem_seg_shape == image_shape:
                return str(out_sem_seg_path.relative_to(self.output_dir))

        sem_seg = self.xml_converter.to_sem_seg(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if sem_seg is None:
            return None

        sem_seg_dir.mkdir(parents=True, exist_ok=True)

        save_image_array_to_path(out_sem_seg_path, sem_seg)

        return str(out_sem_seg_path.relative_to(self.output_dir))

    def save_instances(
        self, xml_path: Path, image_stem: str, original_image_shape: tuple[int, int], image_shape: tuple[int, int]
    ):
        if self.output_dir is None:
            raise ValueError("Cannot run when the output dir is not set")
        instances_dir = self.output_dir.joinpath("instances")
        out_instances_path = instances_dir.joinpath(image_stem + ".json")
        out_instances_size_path = instances_dir.joinpath(image_stem + ".txt")

        # Check if image already exist and if it doesn't need resizing
        if not self.overwrite and out_instances_path.exists() and out_instances_size_path.exists():
            with out_instances_size_path.open(mode="r") as f:
                out_intstances_shape = tuple(int(x) for x in f.read().strip().split(","))
            if out_intstances_shape == image_shape:
                return str(out_instances_path.relative_to(self.output_dir))

        instances = self.xml_converter.to_instances(
            xml_path, original_image_shape=original_image_shape, image_shape=image_shape
        )
        if instances is None:
            return None

        instances_dir.mkdir(parents=True, exist_ok=True)

        json_instances = {"image_size": image_shape, "annotations": instances}
        with out_instances_path.open(mode="w") as f:
            json.dump(json_instances, f)
        with out_instances_size_path.open(mode="w") as f:
            f.write(f"{image_shape[0]},{image_shape[1]}")

        return str(out_instances_path.relative_to(self.output_dir))

    def save_panos(
        self,
        xml_path: Path,
        image_stem: str,
        original_image_shape: tuple[int, int],
        image_shape: tuple[int, int],
    ):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output dir is None")
        panos_dir = self.output_dir.joinpath("panos")

        out_pano_path = panos_dir.joinpath(image_stem + ".png")
        out_segments_info_path = panos_dir.joinpath(image_stem + ".json")

        # Check if image already exist and if it doesn't need resizing
        if not self.overwrite and out_pano_path.exists():
            out_pano_shape = imagesize.get(out_pano_path)[::-1]
            if out_pano_shape == image_shape:
                return str(out_pano_path.relative_to(self.output_dir)), str(out_segments_info_path.relative_to(self.output_dir))

        pano_output = self.xml_converter.to_pano(xml_path, original_image_shape=original_image_shape, image_shape=image_shape)
        if pano_output is None:
            return None
        pano, segments_info = pano_output

        panos_dir.mkdir(parents=True, exist_ok=True)

        save_image_array_to_path(out_pano_path, pano)

        json_pano = {"image_size": image_shape, "segments_info": segments_info}
        with out_segments_info_path.open(mode="w") as f:
            json.dump(json_pano, f)

        return str(out_pano_path.relative_to(self.output_dir)), str(out_segments_info_path.relative_to(self.output_dir))

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
        xml_path = image_path_to_xml_path(image_path, self.disable_check)
        # xml_path = self.input_dir.joinpath("page", image_stem + '.xml')

        original_image_shape = tuple(int(value) for value in imagesize.get(image_path)[::-1])
        original_image_dpi = imagesize.getDPI(image_path)
        assert len(original_image_dpi) == 2, f"Invalid DPI: {original_image_dpi}"
        assert original_image_dpi[0] == original_image_dpi[1], f"Non-square DPI: {original_image_dpi}"
        original_image_dpi = original_image_dpi[0]
        image_shape = self.augmentations[0].get_output_shape(
            original_image_shape[0], original_image_shape[1], dpi=original_image_dpi
        )

        results = {}
        results["original_image_paths"] = str(image_path)

        out_image_path = self.save_image(image_path, image_stem, original_image_shape, image_shape)
        if out_image_path is not None:
            results["image_paths"] = out_image_path

        out_sem_seg_path = self.save_sem_seg(xml_path, image_stem, original_image_shape, image_shape)
        if out_sem_seg_path is not None:
            results["sem_seg_paths"] = out_sem_seg_path

        out_instances_path = self.save_instances(xml_path, image_stem, original_image_shape, image_shape)
        if out_instances_path is not None:
            results["instances_paths"] = out_instances_path

        pano_output = self.save_panos(xml_path, image_stem, original_image_shape, image_shape)
        if pano_output is not None:
            out_pano_path, out_segments_info_path = pano_output
            results["pano_paths"] = out_pano_path
            results["segments_info_paths"] = out_segments_info_path

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
            if mode != self.xml_converter.xml_regions.mode:
                self.overwrite = True

        with mode_path.open(mode="w") as f:
            f.write(self.xml_converter.xml_regions.mode)

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
            "classes": self.xml_converter.xml_regions.regions,
            "mode": self.xml_converter.xml_regions.mode,
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
    xml_converter = XMLConverter(xml_regions, args.square_lines)

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
        xml_converter=xml_converter,
        disable_check=args.disable_check,
        overwrite=args.overwrite,
    )
    process.run()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
