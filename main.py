import argparse
from typing import List, Optional
import torch


from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.model_zoo import get, get_checkpoint_url
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.config import CfgNode
from datasets.dataset_v2 import dataset_dict_loader
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.events import EventStorage
from detectron2.data import transforms as T
from detectron2.data.transforms import Augmentation, Transform


from datasets.transforms_v2 import RandomElastic, Affine, RandomFlip

# TODO Replace with LazyConfig

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")
    
    detectron2_args = parser.add_argument_group("detectron2")
    
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])
    
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
    

def setup_cfg(args, cfg: Optional[CfgNode]=None) -> CfgNode:
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # cfg.MODEL.ROI_HEADS.CLS_AGNOSTIC_MASK = True
    
    cfg.freeze()
    
    return cfg

def build_augmentation(cfg, is_train) -> List[Augmentation | Transform]:
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation: List[Augmentation | Transform] = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    
    if not is_train:
        return augmentation
    
    if cfg.INPUT.RANDOM_FLIP != "none":    
        if cfg.INPUT.RANDOM_FLIP == "horizontal" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(
                RandomFlip(
                    horizontal=True,
                    vertical=False,
                )
            )
        if cfg.INPUT.RANDOM_FLIP == "vertical" or cfg.INPUT.RANDOM_FLIP == "both":
            augmentation.append(
                RandomFlip(
                    horizontal=False,
                    vertical=True,
                )
            )
        
    augmentation.append(RandomElastic(prob=0.5, alpha=34, stdv=4))
    # print(augmentation)
    return augmentation
    
class Trainer(DefaultTrainer):
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type =="sem_seg":
            evaluator = SemSegEvaluator(
                dataset_name=dataset_name,
                distributed=True,
                output_dir=cfg.OUTPUT_DIR
            )
        else:
            raise NotImplementedError(f"Current evaluator type {evaluator_type} not supported")
        
        return evaluator
    
    @classmethod    
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(is_train=True, 
                                   augmentations=build_augmentation(cfg, is_train=True), 
                                   image_format=cfg.INPUT.FORMAT,
                                   use_instance_mask=cfg.MODEL.MASK_ON,
                                   instance_mask_format=cfg.INPUT.MASK_FORMAT,
                                   use_keypoint=cfg.MODEL.KEYPOINT_ON)
        else:
            raise NotImplementedError(f"Current META_ARCHITECTURE type {cfg.MODEL.META_ARCHITECTURE} not supported")
        
        return build_detection_train_loader(cfg, mapper=mapper)

def main(args):
    
    # get("../configs/Misc/semantic_R_50_FPN_1x.yaml")
    
    cfg = setup_cfg(args)
    
    DatasetCatalog.register(
        name="pagexml_train",
        func=lambda path=args.train: dataset_dict_loader(path)
    )
    MetadataCatalog.get("pagexml_train").set(stuff_classes=["backgroud", "baseline"])
    MetadataCatalog.get("pagexml_train").set(stuff_colors=[(0,0,0), (255,255,255)])
    MetadataCatalog.get("pagexml_train").set(evaluator_type="sem_seg")
    MetadataCatalog.get("pagexml_train").set(ignore_label=255)
    
    DatasetCatalog.register(
        name="pagexml_val",
        func=lambda path=args.val: dataset_dict_loader(path)
    )
    MetadataCatalog.get("pagexml_val").set(stuff_classes=["backgroud", "baseline"])
    MetadataCatalog.get("pagexml_val").set(stuff_colors=[(0,0,0), (255,255,255)])
    MetadataCatalog.get("pagexml_val").set(evaluator_type="sem_seg")
    MetadataCatalog.get("pagexml_val").set(ignore_label=255)
    
    trainer = Trainer(cfg=cfg)
    trainer.resume_or_load(resume=False)
    
    # print(trainer.model)
    
    with EventStorage() as storage:
        trainer.train()
    
if __name__ == "__main__":
    args = get_arguments()
    main(args)