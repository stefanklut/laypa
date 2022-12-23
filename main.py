import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence
import torch

from datasets import dataset
from configs.defaults import _C as _C_default
from configs.extra_defaults import _C as _C_extra


from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper
)
from detectron2.utils import comm
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import (
    get_cfg,
    CfgNode
)
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.env import seed_all_rng
from detectron2.data import transforms as T
from detectron2.engine import hooks, launch


from datasets.augmentations import build_augmentation
from datasets.preprocess import Preprocess
from utils.path_utils import unique_path
from page_xml.xml_to_image import XMLImage
from utils.tempdir import OptionalTemporaryDirectory

import models

# TODO Replace with LazyConfig

# torch.autograd.set_detect_anomaly(True)

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main file for Layout Analysis")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument(
        "-c", "--config", help="config file", required=True)
    detectron2_args.add_argument(
        "--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file",
                            nargs="+", action="extend", type=str)
    io_args.add_argument("-v", "--val", help="Validation input folder/file",
                            nargs="+", action="extend", type=str)
    
    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument(
        "--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument(
        "--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")
    
    # other_args.add_argument("--img_list", help="List with location of images")
    # other_args.add_argument("--label_list", help="List with location of labels")
    # other_args.add_argument("--out_size_list", help="List with sizes of images")
    
    # From detectron2.engine.defaults
    gpu_args = parser.add_argument_group("GPU launch")
    gpu_args.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    gpu_args.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    gpu_args.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    gpu_args.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    args = parser.parse_args()

    return args


def setup_cfg(args, cfg: Optional[CfgNode] = None, save_config=True) -> CfgNode:
    """
    Create the config used for training and evaluation. 
    Loads from default configs and merges with specific config file specified in the command line arguments

    Args:
        args (argparse.Namespace): arguments used to load a config file, also used for overwriting values directly (--opts)
        cfg (Optional[CfgNode], optional): possible overwrite of default config. Defaults to None.
        save_config (bool, optional): flag whether or not to save the config (should be True during training). Defaults to True.

    Returns:
        CfgNode: config
    """
    if cfg is None:
        cfg = _C_default

    # Merge with extra defaults, config file and command line args
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(_C_extra)
    cfg.set_new_allowed(False)

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    
    if not cfg.SEED < 0:
        seed = cfg.SEED
        rank = comm.get_rank()
        seed_all_rng(None if seed < 0 else seed + rank)
        # TODO What to do with this, NLL is not deterministic. Only during training
        # torch.use_deterministic_algorithms(True)
    
    # For saving/documentation purposes
    now = datetime.now()
    formatted_datetime = f"{now:%Y-%m-%d_%H-%M-%S}"
    cfg.SETUP_TIME = formatted_datetime
    
    cfg.CONFIG_PATH = str(Path(args.config).resolve())
    
    # Setup run specific folders to prevent overwrites
    if cfg.OUTPUT_DIR and (cfg.RUN_DIR or cfg.NAME) and not cfg.MODEL.RESUME:
        folder_name = []
        if cfg.RUN_DIR:
            folder_name.append(f"RUN_{formatted_datetime}")
        if cfg.NAME:
            folder_name.append(cfg.NAME)
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "_".join(folder_name))
    
    
    if cfg.MODEL.DEVICE:
        # If cuda device can not be found, default to cpu
        if torch.cuda.device_count() == 0:
            cfg.MODEL.DEVICE = 'cpu'
    else:
        # If not specified use cuda if possible
        if torch.cuda.device_count() > 0:
            cfg.MODEL.DEVICE = 'cuda'
        else:
            cfg.MODEL.DEVICE = 'cpu'

    cfg.freeze()
    
    # Save the confic with all (changed) parameters to a yaml
    if comm.is_main_process() and cfg.OUTPUT_DIR and save_config:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg_output_path = Path(cfg.OUTPUT_DIR).joinpath("config.yaml")
        cfg_output_path = unique_path(cfg_output_path)
        with open(cfg_output_path, mode="w") as f:
            f.write(cfg.dump())
    return cfg

def preprocess_datasets(cfg: CfgNode, 
                        train: str | Path | Sequence[str|Path], 
                        val: str | Path | Sequence[str|Path], 
                        output_dir: str | Path):
    """
    Preprocess the dataset(s). Converts ground truth pageXML to label masks for training

    Args:
        cfg (CfgNode): config
        train (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the training images
        val (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the validation images
        output_dir (str | Path): path to output dir, where the processed data will be saved

    Raises:
        FileNotFoundError: a training dir/txt does not exist
        FileNotFoundError: a validation dir/txt does not exist
        FileNotFoundError: the output dir does not exist
    """
    
    train = Preprocess.clean_input_paths(train)
    val = Preprocess.clean_input_paths(val)
        
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        
    if not all(missing := path.exists() for path in train):
        raise FileNotFoundError(f"Train File/Folder not found: {missing} does not exist")
    
    if not all(missing := path.exists() for path in val):
        raise FileNotFoundError(f"Validation File/Folder not found: {missing} does not exist")
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output Folder not found: {output_dir} does not exist")
    
    xml_to_image = XMLImage(
        mode=cfg.MODEL.MODE,
        line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
        line_color=cfg.PREPROCESS.BASELINE.LINE_COLOR,
        regions=cfg.PREPROCESS.REGION.REGIONS,
        merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
        region_type=cfg.PREPROCESS.REGION.REGION_TYPE
    )
    
    process = Preprocess(
        input_paths=None,
        output_dir=None,
        resize=cfg.PREPROCESS.RESIZE.USE,
        resize_mode=cfg.PREPROCESS.RESIZE.RESIZE_MODE,
        min_size=cfg.PREPROCESS.RESIZE.MIN_SIZE,
        max_size=cfg.PREPROCESS.RESIZE.MAX_SIZE,
        xml_to_image=xml_to_image,
        disable_check=cfg.PREPROCESS.DISABLE_CHECK,
        overwrite=cfg.PREPROCESS.OVERWRITE
    )
    
    # Train
    train_output_dir = output_dir.joinpath('train')
    process.set_input_paths(train)
    process.set_output_dir(train_output_dir)
    process.run()
    
    # Validation
    val_output_dir = output_dir.joinpath('val')
    process.set_input_paths(val)
    process.set_output_dir(val_output_dir)
    process.run()
    
    dataset.register_dataset(train_output_dir, 
                             val_output_dir, 
                             train_name="train", 
                             val_name="val", 
                             mode=cfg.MODEL.MODE)

class Trainer(DefaultTrainer):
    """
    Trainer class
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.checkpointer.save_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        os.makedirs(self.checkpointer.save_dir, exist_ok=True)
        
        
        miou_checkpointer = hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                                   checkpointer=self.checkpointer,
                                                   val_metric="sem_seg/mIoU",
                                                   mode='max',
                                                   file_prefix='model_best_mIoU')
        
        fwiou_checkpointer = hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                                   checkpointer=self.checkpointer,
                                                   val_metric="sem_seg/fwIoU",
                                                   mode='max',
                                                   file_prefix='model_best_fwIoU')
        
        macc_checkpointer = hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                                   checkpointer=self.checkpointer,
                                                   val_metric="sem_seg/mACC",
                                                   mode='max',
                                                   file_prefix='model_best_mACC')
        
        pacc_checkpointer = hooks.BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                                                   checkpointer=self.checkpointer,
                                                   val_metric="sem_seg/pACC",
                                                   mode='max',
                                                   file_prefix='model_best_pACC')
        
        self.register_hooks([miou_checkpointer, 
                             fwiou_checkpointer, 
                             macc_checkpointer, 
                             pacc_checkpointer])

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

def setup_training(args):
    """
    Setup and start training

    Args:
        args (argparse.Namespace): arguments used to load a config file, also used for overwriting values directly (--opts)

    Returns:
        OrderedDict|None: results, if evaluation is enabled. Otherwise None.
    """
    cfg = setup_cfg(args)

    # Temp dir for preprocessing in case no temporary dir was specified
    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not(args.keep_tmp_dir)) as tmp_dir:
        
        preprocess_datasets(cfg, args.train, args.val, tmp_dir)
    
        trainer = Trainer(cfg=cfg)
        if not cfg.TRAIN.WEIGHTS:
            trainer.resume_or_load(resume=cfg.MODEL.RESUME)
        else:
            trainer.checkpointer.load(cfg.TRAIN.WEIGHTS)
            if trainer.checkpointer.has_checkpoint():
                trainer.start_iter = trainer.iter + 1

        # print(trainer.model)
        
        results = trainer.train()

    return results

def main(args) -> None:
    assert args.num_gpus <= torch.cuda.device_count(), \
        f"Less GPUs found ({torch.cuda.device_count()}) than specified ({args.num_gpus})"
    
    launch(
        setup_training,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
    

if __name__ == "__main__":
    args = get_arguments()
    main(args)
    
    
