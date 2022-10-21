# Based on detectron2.evaluation.SemSegEvaluator

import argparse
from collections import OrderedDict
from multiprocessing import Pool, Value
import os
from pathlib import Path
import numpy as np
import cv2
from detectron2.data import Metadata
from typing import Optional
import logging
from natsort import os_sorted
from page_xml.xml_to_image import XMLImage
from tqdm import tqdm
from utils.path import clean_input

def get_arguments() -> argparse.Namespace:
    
    # HACK hardcoded regions if none are given
    republic_regions = ["marginalia", "page-number", "resolution", "date",
                        "index", "attendance", "Resumption", "resumption", "Insertion", "insertion"]
    republic_merge_regions = [
        "resolution:Resumption,resumption,Insertion,insertion"]
    parser = argparse.ArgumentParser(
        description="Preprocessing an annotated dataset of documents with pageXML")
    
    parser.add_argument("-g", "--gt", help="Input folder/files GT", nargs="+", default=[],
                        required=True, type=str)
    parser.add_argument("-i", "--input", help="Input folder/files", nargs="+", default=[],
                        required=True, type=str)
    
    parser.add_argument("-m", "--mode", help="Output mode",
                        choices=["baseline", "region", "both"], default="baseline", type=str)

    parser.add_argument("-w", "--line_width",
                        help="Used line width", type=int, default=5)
    parser.add_argument("-c", "--line_color", help="Used line color",
                        choices=list(range(256)), type=int, metavar="{0-255}", default=1)

    parser.add_argument(
        "--regions",
        default=republic_regions,
        nargs="+",
        type=str,
        help="""List of regions to be extracted. 
                            Format: --regions r1 r2 r3 ...""",
    )
    parser.add_argument(
        "--merge_regions",
        default=republic_merge_regions,
        nargs="+",
        type=str,
        help="""Merge regions on PAGE file into a single one.
                            Format --merge_regions r1:r2,r3 r4:r5, then r2 and r3
                            will be merged into r1 and r5 into r4""",
    )
    parser.add_argument(
        "--region_type",
        default=None,
        nargs="+",
        type=str,
        help="""Type of region on PAGE file.
                            Format --region_type t1:r1,r3 t2:r5, then type t1
                            will assigned to regions r1 and r3 and type t2 to
                            r5 and so on...""",
    )

    args = parser.parse_args()
    return args

class XMLEvaluator:
    def __init__(self, 
                 metadata: Optional[Metadata] = None, 
                 ignore_label: Optional[int] = None,
                 class_names: Optional[list[str]] = None) -> None:
        
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
        
        self._logger = logging.getLogger(__name__)

        self.reset()

    def reset(self):
        if self._num_classes is None:
            raise ValueError
        
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )

    def process(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        if self._conf_matrix is None:
            raise ValueError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise ValueError("Must set/reset the boundry confusion matrix")
        if self._num_classes is None:
            raise ValueError
        
        for input_i, output_i in zip(inputs, outputs):

            input_i[input_i == self._ignore_label] = self._num_classes

            _conf_matrix = np.bincount(
                (self._num_classes + 1) *
                output_i.reshape(-1) + input_i.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._conf_matrix += _conf_matrix
            
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(input_i.astype(np.uint8))
                b_pred = self._mask_to_boundary(output_i.astype(np.uint8))

                _b_conf_matrix = np.bincount(
                    (self._num_classes + 1) *
                    b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
                
                self._b_conf_matrix += _b_conf_matrix
                
    def process_output(self, inputs: list[np.ndarray], outputs: list[np.ndarray]):
        # Does not update the internal confusion matrix
        if self._conf_matrix is None:
            raise ValueError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise ValueError("Must set/reset the boundry confusion matrix")
        if self._num_classes is None:
            raise ValueError
        
        full_conf_matrix = np.zeros_like(self._b_conf_matrix)
        full_b_conf_matrix = np.zeros_like(self._b_conf_matrix)
        
        for input_i, output_i in zip(inputs, outputs):

            input_i[input_i == self._ignore_label] = self._num_classes

            _conf_matrix = np.bincount(
                (self._num_classes + 1) *
                output_i.reshape(-1) + input_i.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
            
            full_conf_matrix += _conf_matrix
            
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(input_i.astype(np.uint8))
                b_pred = self._mask_to_boundary(output_i.astype(np.uint8))

                _b_conf_matrix = np.bincount(
                    (self._num_classes + 1) *
                    b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)
                
                full_b_conf_matrix += _b_conf_matrix
        
        return full_conf_matrix, full_b_conf_matrix

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._conf_matrix is None:
            raise ValueError("Must set/reset the confusion matrix")
        if self._b_conf_matrix is None:
            raise ValueError("Must set/reset the boundry confusion matrix")
        if self._num_classes is None:
            raise ValueError("Must set number of classes")
        if self._class_names is None:
            raise ValueError("Must set class names")
        
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
            b_pos_gt = np.sum(
                self._b_conf_matrix[:-1, :-1], axis=0).astype(np.float64)
            b_pos_pred = np.sum(
                self._b_conf_matrix[:-1, :-1], axis=1).astype(np.float64)
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
        self._logger.info(results)
        print(results)
        return results

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(
            mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(
            padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary

class EvalWrapper:
    def __init__(self, xml_to_image: XMLImage, evaluator: XMLEvaluator) -> None:
        self.xml_to_image: XMLImage = xml_to_image
        self.evaluator: XMLEvaluator = evaluator
    
    def compare_xml(self, info: tuple[Path, Path]):
        xml_i_1, xml_i_2 = info
        
        if xml_i_1.stem != xml_i_2.stem:
            raise ValueError(f"XMLs {xml_i_1} & {xml_i_2} do not match")
    
        image_i_1 = self.xml_to_image.run(xml_i_1)
        image_i_2 = self.xml_to_image.run(xml_i_2)
        
        self.evaluator.process([image_i_1], [image_i_2])
    
    def compare_xml_output(self, info: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
        xml_i_1, xml_i_2 = info
        
        if xml_i_1.stem != xml_i_2.stem:
            raise ValueError(f"XMLs {xml_i_1} & {xml_i_2} do not match")
    
        image_i_1 = self.xml_to_image.run(xml_i_1)
        image_i_2 = self.xml_to_image.run(xml_i_2)
        
        confusion_matrix  = self.evaluator.process_output([image_i_1], [image_i_2])
        
        return confusion_matrix
    
    def compare_images(self, info: tuple[Path, Path]):
        image_path_i_1, image_path_i_2 = info
        
        if image_path_i_1.stem != image_path_i_2.stem:
            raise ValueError(f"Images {image_path_i_1} & {image_path_i_2} do not match")
    
        image_i_1 = cv2.imread(image_path_i_1, cv2.IMREAD_GRAYSCALE)
        image_i_2 = cv2.imread(image_path_i_2, cv2.IMREAD_GRAYSCALE)
        
        self.evaluator.process([image_i_1], [image_i_2])
    
    def compare_images_output(self, info: tuple[Path, Path]) -> tuple[np.ndarray, np.ndarray]:
        image_path_i_1, image_path_i_2 = info
        
        if image_path_i_1.stem != image_path_i_2.stem:
            raise ValueError(f"Images {image_path_i_1} & {image_path_i_2} do not match")
    
        image_i_1 = cv2.imread(image_path_i_1, cv2.IMREAD_GRAYSCALE)
        image_i_2 = cv2.imread(image_path_i_2, cv2.IMREAD_GRAYSCALE)
        
        confusion_matrix = self.evaluator.process_output([image_i_1], [image_i_2])
        
        return confusion_matrix
    
    def run_xml(self, xml_list1: list[Path], xml_list2: list[Path]):
        if len(xml_list1) != len(xml_list2):
            raise ValueError("Number of xml files does not match")
        
        # #Single thread
        # for xml_i_1, xml_i_2 in tqdm(zip(xml_list1, xml_list2), total=len(xml_list1)):
        #     self.compare_xml((xml_i_1, xml_i_2))
        
        
        # Multi thread
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(
                self.compare_xml_output, list(zip(xml_list1, xml_list2))), total=len(xml_list1)))
        
        results = np.asarray(results)
        self.evaluator._conf_matrix += np.sum(results[:, 0], axis=0)
        self.evaluator._b_conf_matrix += np.sum(results[:, 1], axis=0)
        
    def run_images(self, image_path_list1: list[Path], image_path_list2: list[Path]):
        if len(image_path_list1) != len(image_path_list2):
            raise ValueError("Number of image paths does not match")
        # #Single thread
        # for image_path_i_1, image_path_i_2 in tqdm(zip(image_path_list1, image_path_list2), total=len(image_path_list1)):
        #     self.compare_images((image_path_i_1, image_path_i_2))
        
        
        # Multi thread
        with Pool(os.cpu_count()) as pool:
            results = list(tqdm(pool.imap_unordered(
                self.compare_images_output, list(zip(image_path_list1, image_path_list2))), total=len(image_path_list1)))
        
        results = np.asarray(results)
        self.evaluator._conf_matrix += np.sum(results[:, 0], axis=0)
        self.evaluator._b_conf_matrix += np.sum(results[:, 1], axis=0)
    
    def evaluate(self):
        return self.evaluator.evaluate()

def main(args):
    xml_list1 = clean_input(args.gt, suffixes=[".xml"])
    xml_list2 = clean_input(args.input, suffixes=[".xml"])
    
    xml_to_image = XMLImage(
        mode=args.mode,
        line_width=args.line_width,
        line_color=args.line_color,
        regions=args.regions,
        merge_regions=args.merge_regions,
        region_type=args.region_type
    )
    
    evaluator = XMLEvaluator(ignore_label=255, 
                             class_names=xml_to_image.get_regions())
    
    eval_runner = EvalWrapper(xml_to_image, evaluator)
    eval_runner.run_xml(xml_list1, xml_list2)
    eval_runner.evaluate()

if __name__ == "__main__":
    args = get_arguments()
    main(args)