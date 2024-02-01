# Based on detectron2.evaluation.SemSegEvaluator

import argparse

# from multiprocessing.pool import ThreadPool as Pool
import os
from collections import OrderedDict
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from detectron2.data import Metadata
from tqdm import tqdm

from page_xml.xml_converter import XMLConverter
from page_xml.xml_regions import XMLRegions
from utils.input_utils import get_file_paths


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        parents=[XMLRegions.get_parser()], description="Compare two sets of pagexml, based on pixel level metrics"
    )

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-g", "--gt", help="Input folder/files GT", nargs="+", default=[], required=True, type=str, action="extend"
    )
    io_args.add_argument(
        "-i", "--input", help="Input folder/files", nargs="+", default=[], required=True, type=str, action="extend"
    )

    xml_converter_args = parser.add_argument_group("XML Converter")
    xml_converter_args.add_argument("--square-lines", help="Square the lines", action="store_true")

    args = parser.parse_args()
    return args


class IOUEvaluator:
    """
    Class for saving IOU results
    """

    def __init__(
        self,
        metadata: Optional[Metadata] = None,
        ignore_label: Optional[int] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        """
        Class for saving IOU results

        Args:
            metadata (Optional[Metadata], optional): if available get class info from metadata. Defaults to None.
            ignore_label (Optional[int], optional): ignored label. Defaults to None.
            class_names (Optional[list[str]], optional): names for each class in the prediction. Defaults to None.
        """

        self._class_names: Optional[list[str]] = None
        self._num_classes: Optional[int] = None
        self._ignore_label: Optional[int] = None

        if metadata is not None:
            self._class_names = class_names if class_names is not None else metadata.stuff_classes
            self._num_classes = len(class_names) if class_names is not None else len(metadata.stuff_classes)
            self._ignore_label = ignore_label if ignore_label is not None else metadata.ignore_label
        else:
            assert class_names is not None
            assert ignore_label is not None

            self._class_names = class_names
            self._num_classes = len(class_names)
            self._ignore_label = ignore_label

        self._compute_boundary_iou = True

        self._conf_matrix = None
        self._b_conf_matrix = None

        self.reset()

    def reset(self):
        """
        Reset the internal confusion matrices

        Raises:
            TypeError: number of classes has not been set
        """
        if self._num_classes is None:
            raise TypeError("Must set number of classes")

        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)

    def process(
        self,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
    ) -> None:
        """
        Update the internal confusion matrix

        Args:
            inputs (list[np.ndarray]): array of ground truth
            outputs (list[np.ndarray]): array of predictions

        Raises:
            TypeError: confusion matrix has not been initialized
            TypeError: boundary confusion matrix has not been initialized
            TypeError: number of classes has not been set
        """
        if self._conf_matrix is None:
            raise TypeError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise TypeError("Must set/reset the boundary confusion matrix")
        if self._num_classes is None:
            raise TypeError("Must set number of classes")

        for input_i, output_i in zip(inputs, outputs):
            input_i[input_i == self._ignore_label] = self._num_classes

            _conf_matrix = np.bincount(
                (self._num_classes + 1) * output_i.reshape(-1) + input_i.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._conf_matrix += _conf_matrix

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(input_i.astype(np.uint8))
                b_pred = self._mask_to_boundary(output_i.astype(np.uint8))

                _b_conf_matrix = np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

                self._b_conf_matrix += _b_conf_matrix

    def process_output(
        self,
        inputs: list[np.ndarray],
        outputs: list[np.ndarray],
    ):
        """
        Output a confusion matrix for the processing externally

        Args:
            inputs (list[np.ndarray]): array of ground truth
            outputs (list[np.ndarray]): array of predictions

        Raises:
            TypeError: confusion matrix has not been initialized
            TypeError: boundary confusion matrix has not been initialized
            TypeError: number of classes has not been set
        """
        # Does not update the internal confusion matrix
        if self._conf_matrix is None:
            raise TypeError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise TypeError("Must set/reset the boundary confusion matrix")
        if self._num_classes is None:
            raise TypeError("Must set number of classes")

        full_conf_matrix = np.zeros_like(self._b_conf_matrix)
        full_b_conf_matrix = np.zeros_like(self._b_conf_matrix)

        for input_i, output_i in zip(inputs, outputs):
            input_i[input_i == self._ignore_label] = self._num_classes

            _conf_matrix = np.bincount(
                (self._num_classes + 1) * output_i.reshape(-1) + input_i.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            full_conf_matrix += _conf_matrix

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(input_i.astype(np.uint8))
                b_pred = self._mask_to_boundary(output_i.astype(np.uint8))

                _b_conf_matrix = np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

                full_b_conf_matrix += _b_conf_matrix

        return full_conf_matrix, full_b_conf_matrix

    def evaluate(self):
        # TODO Change the variable names to make it clearer what is being calculated
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._conf_matrix is None:
            raise TypeError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise TypeError("Must set/reset the boundry confusion matrix")
        if self._num_classes is None:
            raise TypeError("Must set number of classes")
        if self._class_names is None:
            raise TypeError("Must set class names")

        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=np.float64)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(np.float64)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float64)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float64)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        results = OrderedDict({"sem_seg": res})
        return results

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary


class EvalWrapper:
    """
    Wrapper to run the IOUEvaluator with pageXML, requires conversion to mask images first
    """

    def __init__(self, xml_to_image: XMLConverter, evaluator: IOUEvaluator) -> None:
        """
        Wrapper to run the IOUEvaluator with pageXML, requires conversion to mask images first

        Args:
            xml_to_image (XMLImage): converter from pageXML to mask image
            evaluator (IOUEvaluator): evaluator for the XML
        """
        self.xml_to_image: XMLConverter = xml_to_image
        self.evaluator: IOUEvaluator = evaluator

    def compare_xml(self, info: tuple[Path, Path]):
        """
        Conversion of two pageXMLs to a mask, and then running these masks through the IOUEvaluator

        Args:
            info (tuple[Path, Path]): (tuple containing)
                Path: ground truth pageXML path
                Path: predicted pageXML path

        Raises:
            ValueError: the names of the pageXMLs do not match
        """
        xml_i_1, xml_i_2 = info

        if xml_i_1.stem != xml_i_2.stem:
            raise ValueError(f"XMLs {xml_i_1} & {xml_i_2} do not match")

        image_i_1 = self.xml_to_image.to_sem_seg(xml_i_1)
        image_i_2 = self.xml_to_image.to_sem_seg(xml_i_2)

        self.evaluator.process([image_i_1], [image_i_2])

    def compare_xml_output(self, info: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
        """
        Conversion of two pageXMLs to a mask, and then running these masks through the IOUEvaluator
        With the return option multiprocessing is possible, no internal values are overwritten

        Args:
            info (tuple[Path, Path]): (tuple containing)
                Path: ground truth pageXML path
                Path: predicted pageXML path

        Raises:
            ValueError: the names of the pageXMLs do not match

        Returns:
            np.ndarray: confusion matrix
            np.ndarray: boundary confusion matrix
        """
        xml_i_1, xml_i_2 = info

        if xml_i_1.stem != xml_i_2.stem:
            raise ValueError(f"XMLs {xml_i_1} & {xml_i_2} do not match")

        image_i_1 = self.xml_to_image.to_sem_seg(xml_i_1)
        image_i_2 = self.xml_to_image.to_sem_seg(xml_i_2)

        confusion_matrix = self.evaluator.process_output([image_i_1], [image_i_2])

        return confusion_matrix

    def compare_images(self, info: tuple[Path, Path]):
        """
        Load two images to a mask, and then running these masks through the IOUEvaluator

        Args:
            info (tuple[Path, Path]): (tuple containing)
                Path: ground truth image path
                Path: predicted image path

        Raises:
            ValueError: the names of the pageXMLs do not match
        """
        image_path_i_1, image_path_i_2 = info

        if image_path_i_1.stem != image_path_i_2.stem:
            raise ValueError(f"Images {image_path_i_1} & {image_path_i_2} do not match")

        image_i_1 = cv2.imread(str(image_path_i_1), cv2.IMREAD_GRAYSCALE)
        image_i_2 = cv2.imread(str(image_path_i_2), cv2.IMREAD_GRAYSCALE)

        self.evaluator.process([image_i_1], [image_i_2])

    def compare_images_output(self, info: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
        """
        Load two images to a mask, and then running these masks through the IOUEvaluator
        With the return option multiprocessing is possible, no internal values are overwritten

        Args:
            info (tuple[Path, Path]): (tuple containing)
                Path: ground truth image path
                Path: predicted image path

        Raises:
            ValueError: the names of the images do not match

        Returns:
            np.ndarray: confusion matrix
            np.ndarray: boundary confusion matrix
        """
        image_path_i_1, image_path_i_2 = info

        if image_path_i_1.stem != image_path_i_2.stem:
            raise ValueError(f"Images {image_path_i_1} & {image_path_i_2} do not match")

        image_i_1 = cv2.imread(str(image_path_i_1), cv2.IMREAD_GRAYSCALE)
        image_i_2 = cv2.imread(str(image_path_i_2), cv2.IMREAD_GRAYSCALE)

        confusion_matrix = self.evaluator.process_output([image_i_1], [image_i_2])

        return confusion_matrix

    def run_xml(self, xml_list1: list[Path], xml_list2: list[Path]):
        """
        Run the xml IOU evaluation on list of pageXML paths

        Args:
            xml_list1 (list[Path]): ground truth pageXML paths
            xml_list2 (list[Path]): predicted pageXML paths

        Raises:
            ValueError: number of xmls in both list is not equal
        """
        if len(xml_list1) != len(xml_list2):
            raise ValueError("Number of xml files does not match")

        # #Single thread
        # for xml_i_1, xml_i_2 in tqdm(zip(xml_list1, xml_list2), total=len(xml_list1)):
        #     self.compare_xml((xml_i_1, xml_i_2))

        # Multi thread
        with Pool(os.cpu_count()) as pool:
            results = list(
                tqdm(pool.imap_unordered(self.compare_xml_output, list(zip(xml_list1, xml_list2))), total=len(xml_list1))
            )

        results = np.asarray(results)
        # HACK to get the confusion matrix back into the evaluator
        self.evaluator._conf_matrix += np.sum(results[:, 0], axis=0)
        self.evaluator._b_conf_matrix += np.sum(results[:, 1], axis=0)

    def run_images(
        self,
        image_path_list1: list[Path],
        image_path_list2: list[Path],
    ):
        """
        Run the IOU evaluation on list of image paths

        Args:
            image_path_list1 (list[Path]): ground truth image paths
            image_path_list2 (list[Path]): predicted image paths

        Raises:
            ValueError: number of images in both list is not equal
        """
        if len(image_path_list1) != len(image_path_list2):
            raise ValueError("Number of image paths does not match")
        # #Single thread
        # for image_path_i_1, image_path_i_2 in tqdm(zip(image_path_list1, image_path_list2), total=len(image_path_list1)):
        #     self.compare_images((image_path_i_1, image_path_i_2))

        # Multi thread
        with Pool(os.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(self.compare_images_output, list(zip(image_path_list1, image_path_list2))),
                    total=len(image_path_list1),
                )
            )

        results = np.asarray(results)
        # HACK to get the confusion matrix back into the evaluator
        self.evaluator._conf_matrix += np.sum(results[:, 0], axis=0)
        self.evaluator._b_conf_matrix += np.sum(results[:, 1], axis=0)

    def evaluate(self):
        """
        Run the full evaluation of the confusion matrix

        Returns:
            OrderDict: values are saved in "sem_seg"
        """
        return self.evaluator.evaluate()


def pretty_print(input_dict: dict[str, float], n_decimals=3):
    """
    Print the dict with better readability

    Args:
        input_dict (dict[str, float]): dictionary of
        n_decimals (int, optional): rounding of the float values. Defaults to 3.
    """
    len_names = max(len(str(key)) for key in input_dict.keys()) + 1
    len_values = max(len(f"{value:.{n_decimals}f}") for value in input_dict.values()) + 1

    output_string = ""
    for key, value in input_dict.items():
        output_string += f"{str(key):<{len_names}}: {value:<{len_values}.{n_decimals}f}\n"

    print(output_string)


def main(args):
    xml_list1 = get_file_paths(args.gt, formats=[".xml"])
    xml_list2 = get_file_paths(args.input, formats=[".xml"])

    # REVIEW Maybe this should also use the config file.
    xml_regions = XMLRegions(
        mode=args.mode,
        line_width=args.line_width,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type,
    )
    xml_to_image = XMLConverter(xml_regions, args.square_lines)

    evaluator = IOUEvaluator(ignore_label=255, class_names=xml_regions.regions)

    eval_runner = EvalWrapper(xml_to_image, evaluator)
    eval_runner.run_xml(xml_list1, xml_list2)
    results = eval_runner.evaluate()["sem_seg"]
    pretty_print(results)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
