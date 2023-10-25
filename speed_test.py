# import argparse
import random
from multiprocessing import Pool

from detectron2.utils.file_io import PathManager
from tqdm import tqdm

from utils.timing_utils import ContextTimer

# from multiprocessing.pool import ThreadPool as Pool
# import cv2
# from pathlib import Path


# import os
# from typing import Optional, Sequence

# from pathlib import Path

# import functools
# import inspect
# import logging
# from fvcore.common.config import CfgNode as _CfgNode

# from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
# from iopath.common.file_io import PathManager as PathManagerBase

# from detectron2.config import CfgNode

# from datasets import dataset
# from datasets.preprocess import Preprocess
# from utils.input_utils import clean_input_paths, get_file_paths
# from page_xml.xml_converter import XMLConverter

# from core.preprocess import preprocess_datasets
# from core.setup import setup_cfg
# from datasets.augmentations import build_augmentation
# from utils.tempdir import OptionalTemporaryDirectory
# from detectron2.data import DatasetMapper, build_detection_train_loader


# def get_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Main file for Layout Analysis")

#     detectron2_args = parser.add_argument_group("detectron2")

#     detectron2_args.add_argument(
#         "-c", "--config", help="config file", required=True)
#     detectron2_args.add_argument(
#         "--opts", nargs="+", help="optional args to change", default=[])

#     io_args = parser.add_argument_group("IO")
#     io_args.add_argument("-t", "--train", help="Train input folder/file",
#                             nargs="+", action="extend", type=str, required=True)
#     io_args.add_argument("-v", "--val", help="Validation input folder/file",
#                             nargs="+", action="extend", type=str, required=True)

#     tmp_args = parser.add_argument_group("tmp files")
#     tmp_args.add_argument(
#         "--tmp_dir", help="Temp files folder", type=str, default=None)
#     tmp_args.add_argument(
#         "--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

#     args = parser.parse_args()

#     return args

# def load(x):
#     cv2.imread(str(x))


def heavy(x):
    return [random.randint(1, 100) * x for _ in range(1000000)]


def main(args):
    # cfg = setup_cfg(args)

    # with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not(args.keep_tmp_dir)) as tmp_dir:

    #     preprocess_datasets(cfg, args.train, args.val, tmp_dir, save_image_locations=False)

    #     mapper = DatasetMapper(is_train=True,
    #                             recompute_boxes=cfg.MODEL.MASK_ON,
    #                             augmentations=build_augmentation(
    #                                 cfg, is_train=True),
    #                             image_format=cfg.INPUT.FORMAT,
    #                             use_instance_mask=cfg.MODEL.MASK_ON,
    #                             instance_mask_format=cfg.INPUT.MASK_FORMAT,
    #                             use_keypoint=cfg.MODEL.KEYPOINT_ON)

    #     dataloader = iter(build_detection_train_loader(cfg=cfg, mapper=mapper))

    #     for i in tqdm(range(100)):
    #         print(i)
    #         with ContextTimer(label="Load"):
    #             next(dataloader)
    # paths = list(Path(args.train[0]).glob("*.jpg"))
    with ContextTimer(), Pool(os.cpu_count()) as pool:
        # for image_path in tqdm(paths):
        #     cv2.imread(str(image_path))
        # _ = list(tqdm(pool.imap_unordered(load, paths), total=len(paths)))
        _ = list(tqdm(pool.imap_unordered(heavy, list(range(100))), total=100))


if __name__ == "__main__":
    # args = get_arguments()
    # import torch
    # os.sched_setaffinity(os.getpid(), list(range(20)))
    # os.system("taskset -p 0xFFFFFFFFFF %d" % os.getpid())
    import multiprocessing
    import os

    print(multiprocessing.cpu_count())

    print(os.sched_getaffinity(0))
    # os.sched_setaffinity(0, {3,4})
    print(os.sched_getaffinity(0))

    # torch.set_num_threads(100)
    main(args=None)
    print(dict(ContextTimer.stats))
    import numpy as np

    print({key: np.mean(value) for key, value in ContextTimer.stats.items()})
    print({key: np.sum(value) for key, value in ContextTimer.stats.items()})
