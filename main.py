import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import List, Optional
import torch

import datasets.dataset as dataset
from configs.extra_defaults import _C


from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper
)
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import (
    get_cfg,
    CfgNode
)
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.events import EventStorage
from detectron2.data import transforms as T
from detectron2.engine import hooks


from datasets.augmentations import (
    RandomElastic,
    RandomAffine,
    Flip,
    RandomTranslation,
    RandomRotation,
    RandomShear,
    RandomScale,
    ResizeShortestEdge,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomGaussianFilter,
    RandomApply,
    Grayscale
)

# TODO Replace with LazyConfig


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main file for Layout Analysis")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument(
        "-c", "--config", help="config file", required=True)
    detectron2_args.add_argument(
        "--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])

    other_args = parser.add_argument_group("other")
    other_args.add_argument("-t", "--train", help="Train input folder",
                            required=True, type=str)
    other_args.add_argument("-v", "--val", help="Validation input folder",
                            required=True, type=str)
    # other_args.add_argument("--img_list", help="List with location of images")
    # other_args.add_argument("--label_list", help="List with location of labels")
    # other_args.add_argument("--out_size_list", help="List with sizes of images")

    args = parser.parse_args()

    return args


def setup_cfg(args, cfg: Optional[CfgNode] = None) -> CfgNode:
    if cfg is None:
        cfg = get_cfg()

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(_C)
    cfg.set_new_allowed(False)

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR and cfg.RUN_DIR and not cfg.MODEL.RESUME:
        now = datetime.now()
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"RUN_({now:%Y-%m-%d|%H:%M:%S})")

    cfg.freeze()
    return cfg


def build_augmentation(cfg, is_train) -> List[T.Augmentation | T.Transform]:
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation: List[T.Augmentation | T.Transform] = [
        ResizeShortestEdge(min_size, max_size, sample_style)]

    if not is_train:
        return augmentation

    if cfg.INPUT.RANDOM_FLIP != "none":
        if cfg.INPUT.RANDOM_FLIP == "horizontal" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(
                RandomApply(Flip(
                    horizontal=True,
                    vertical=False,
                ), prob=0.5)
            )
        if cfg.INPUT.RANDOM_FLIP == "vertical" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(
                RandomApply(Flip(
                    horizontal=False,
                    vertical=True,
                ), prob=0.5)
            )

    # TODO Give these a proper argument in the config
    
    # TODO Add random crop
    # TODO 90 degree rotation
    
    augmentation.append(RandomApply(Grayscale(image_format=cfg.INPUT.FORMAT), 
                                    prob=0.2))
    
    augmentation.append(RandomApply(RandomBrightness(intensity_min=0.5, 
                                                     intensity_max=1.5), 
                                    prob=0.2))
    augmentation.append(RandomApply(RandomContrast(intensity_min=0.5, 
                                                   intensity_max=1.5), 
                                    prob=0.2))
    augmentation.append(RandomApply(RandomSaturation(intensity_min=0.5, 
                                                     intensity_max=1.5), 
                                    prob=0.2))
    augmentation.append(RandomApply(RandomGaussianFilter(min_sigma=0, 
                                                         max_sigma=3), 
                                    prob=0.2))
    
    
    augmentation.append(RandomApply(RandomElastic(alpha=34, 
                                                  stdv=4), 
                                    prob=0.5))
    augmentation.append(RandomApply(RandomAffine(t_stdv=0.02,
                                                 r_kappa=30,
                                                 sh_kappa=20,
                                                 sc_stdv=0.12),
                                    prob=0.5))
    # augmentation.append(RandomTranslation(t_stdv=0.02))
    # augmentation.append(RandomRotation(r_kappa=30))
    # augmentation.append(RandomShear(sh_kappa=20))
    # augmentation.append(RandomScale(sc_stdv=0.12))
    
    
    
    # print(augmentation)
    return augmentation


class Trainer(DefaultTrainer):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.checkpointer.save_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        os.makedirs(self.checkpointer.save_dir, exist_ok=True)
        
        
        best_checkpointer = hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                                   checkpointer=self.checkpointer,
                                                   val_metric="sem_seg/mIoU",
                                                   mode='max',
                                                   file_prefix='model_best_mIOU')
        
        self.register_hooks([best_checkpointer])

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        sem_seg_output_dir = os.path.join(cfg.OUTPUT_DIR, "semantic_segmentation")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg":
            evaluator = SemSegEvaluator(
                dataset_name=dataset_name,
                distributed=True,
                output_dir=sem_seg_output_dir
            )
        else:
            raise NotImplementedError(
                f"Current evaluator type {evaluator_type} not supported")

        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(is_train=True,
                                   augmentations=build_augmentation(
                                       cfg, is_train=True),
                                   image_format=cfg.INPUT.FORMAT,
                                   use_instance_mask=cfg.MODEL.MASK_ON,
                                   instance_mask_format=cfg.INPUT.MASK_FORMAT,
                                   use_keypoint=cfg.MODEL.KEYPOINT_ON)
        else:
            raise NotImplementedError(
                f"Current META_ARCHITECTURE type {cfg.MODEL.META_ARCHITECTURE} not supported")

        return build_detection_train_loader(cfg=cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(is_train=False,
                                   augmentations=build_augmentation(
                                       cfg, is_train=False),
                                   image_format=cfg.INPUT.FORMAT,
                                   use_instance_mask=cfg.MODEL.MASK_ON,
                                   instance_mask_format=cfg.INPUT.MASK_FORMAT,
                                   use_keypoint=cfg.MODEL.KEYPOINT_ON)
        else:
            raise NotImplementedError(
                f"Current META_ARCHITECTURE type {cfg.MODEL.META_ARCHITECTURE} not supported")

        return build_detection_test_loader(cfg=cfg, mapper=mapper, dataset_name=dataset_name)


def main(args) -> None:

    # get("../configs/Misc/semantic_R_50_FPN_1x.yaml")

    cfg = setup_cfg(args)

    if cfg.MODEL.MODE == "baseline":
        dataset.register_baseline(args.train, args.val)
    elif cfg.MODEL.MODE == "region":
        dataset.register_region(args.train, args.val)
    else:
        raise NotImplementedError(
            f"Only have \"baseline\" and \"region\", given {cfg.MODEL.MODE}")
    
    trainer = Trainer(cfg=cfg)
    if cfg.MODEL.RESUME:
        if not cfg.TRAIN.WEIGHTS:
            trainer.resume_or_load(resume=True)
        else:
            trainer.checkpointer.load(cfg.TRAIN.WEIGHTS)
            trainer.start_iter = trainer.iter + 1

    # print(trainer.model)

    with EventStorage() as storage:
        trainer.train()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
