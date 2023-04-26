import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import List, Optional, Sequence
import weakref
import torch
import logging
root_logger = logging.getLogger()

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
from detectron2.engine import (
    DefaultTrainer, 
    TrainerBase,
    SimpleTrainer,
    AMPTrainer,
    create_ddp_model,
    hooks, 
    launch
)
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.env import seed_all_rng
from detectron2.utils.collect_env import collect_env_info
from detectron2.engine.defaults import _highlight

from datasets.augmentations import build_augmentation
from datasets.preprocess import Preprocess
from utils.input_utils import clean_input_paths, get_file_paths
from utils.logging_utils import get_logger_name, setup_logger
from utils.path_utils import unique_path
from page_xml.xml_converter import XMLConverter
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
        "--opts", nargs="+", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file",
                            nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-v", "--val", help="Validation input folder/file",
                            nargs="+", action="extend", type=str, required=True)
    
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


def setup_cfg(args, cfg: Optional[CfgNode] = None) -> CfgNode:
    """
    Create the config used for training and evaluation. 
    Loads from default configs and merges with specific config file specified in the command line arguments

    Args:
        args (argparse.Namespace): arguments used to load a config file, also used for overwriting values directly (--opts)
        cfg (Optional[CfgNode], optional): possible overwrite of default config. Defaults to None.

    Returns:
        CfgNode: config
    """
    if cfg is None:
        cfg = _C_default
        
    cfg.defrost()

    # Merge with extra defaults, config file and command line args
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(_C_extra)
    
    # Allow setting new to allow for deprecated values in the config
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config)
    
    cfg.set_new_allowed(False)
    cfg.merge_from_list(args.opts)
    
    # For saving/documentation purposes
    now = datetime.now()
    formatted_datetime = f"{now:%Y-%m-%d_%H-%M-%S}"
    cfg.SETUP_TIME = formatted_datetime
    
    cfg.CONFIG_PATH = str(Path(args.config).resolve())
    
    if not hasattr(args, 'train'):
        pass 
    elif args.train is None:
        pass
    elif isinstance(args.train, Sequence):
        cfg.TRAINING_PATHS = [str(Path(path).resolve()) for path in args.train]
    else:
        cfg.TRAINING_PATHS = [str(Path(args.train).resolve())]
      
    if not hasattr(args, 'val'):
        pass
    elif args.val is None:
        pass
    elif isinstance(args.val, Sequence):
        cfg.VALIDATION_PATHS = [str(Path(path).resolve()) for path in args.val]
    else:
        cfg.VALIDATION_PATHS = [str(Path(args.val).resolve())]
    
    
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
    return cfg

def setup_logging(cfg=None, args=None, save_log=True):
    rank = comm.get_rank()
    if save_log and cfg is not None:
        output_dir = cfg.OUTPUT_DIR
    else:
        output_dir = None
    root_logger = logging.getLogger()
    logging.getLogger("fvcore")
    logging.getLogger("detectron2")
    
    logger = setup_logger(output_dir, distributed_rank=rank, name=get_logger_name())
    
    for item in root_logger.manager.loggerDict:
        if item.startswith('detectron2') or item.startswith('fvcore'):
            root_logger.manager.loggerDict[item] = logger
            
    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    if args is not None:
        logger.info("Command line arguments: " + str(args))
        if hasattr(args, "config") and args.config != "":
            logger.info(
                "Contents of args.config: {}:\n{}".format(
                    args.config,
                    _highlight(Path(args.config).open("r").read(), args.config),
                )
            )
    return logger
    
def setup_saving(cfg: CfgNode):
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(get_logger_name())
    
    if comm.is_main_process() and output_dir:
        cfg_output_path = output_dir.joinpath("config.yaml")
        cfg_output_path = unique_path(cfg_output_path)
        if isinstance(cfg, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
            with cfg_output_path.open(mode="w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, str(cfg_output_path))
        logger.info("Full config saved to {}".format(cfg_output_path))
        
def setup_seed(cfg: CfgNode):
        seed = cfg.SEED if cfg.SEED else -1
        rank = comm.get_rank()
        seed_all_rng(None if seed < 0 else seed + rank)
        # REVIEW What to do with this, NLL is not deterministic. Only during training
        # torch.use_deterministic_algorithms(True)

def preprocess_datasets(cfg: CfgNode, 
                        train: Optional[str | Path | Sequence[str|Path]], 
                        val: Optional[str | Path | Sequence[str|Path]], 
                        output_dir: str | Path,
                        save_image_locations: bool = True):
    """
    Preprocess the dataset(s). Converts ground truth pageXML to label masks for training

    Args:
        cfg (CfgNode): config
        train (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the training images
        val (str | Path | Sequence[str | Path]): path to dir/txt(s) containing the validation images
        output_dir (str | Path): path to output dir, where the processed data will be saved
        save_image_locations (bool): flag to save processed image locations (for retraining)
        
    Raises:
        FileNotFoundError: a training dir/txt does not exist
        FileNotFoundError: a validation dir/txt does not exist
        FileNotFoundError: the output dir does not exist
    """
    
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output Folder not found: {output_dir} does not exist")
    
    xml_converter = XMLConverter(
        mode=cfg.MODEL.MODE,
        line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
        regions=cfg.PREPROCESS.REGION.REGIONS,
        merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
        region_type=cfg.PREPROCESS.REGION.REGION_TYPE
    )
    
    assert (n_regions := len(xml_converter.get_regions())) == (n_classes := cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES), \
        f"Number of specified regions ({n_regions}) does not match the number of specified classes ({n_classes})"
    
    process = Preprocess(
        input_paths=None,
        output_dir=None,
        resize=cfg.PREPROCESS.RESIZE.USE,
        resize_mode=cfg.PREPROCESS.RESIZE.RESIZE_MODE,
        min_size=cfg.PREPROCESS.RESIZE.MIN_SIZE,
        max_size=cfg.PREPROCESS.RESIZE.MAX_SIZE,
        xml_converter=xml_converter,
        disable_check=cfg.PREPROCESS.DISABLE_CHECK,
        overwrite=cfg.PREPROCESS.OVERWRITE
    )
    
    
    train_output_dir = None
    if train is not None:
        train = clean_input_paths(train)
        if not all(missing := path.exists() for path in train):
            raise FileNotFoundError(f"Train File/Folder not found: {missing} does not exist")
        
        train_output_dir = output_dir.joinpath('train')
        process.set_input_paths(train)
        process.set_output_dir(train_output_dir)
        train_image_paths = get_file_paths(train, process.image_formats)
        process.run()
        
        if save_image_locations:
            #Saving the images used to a txt file
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            train_image_output_path = Path(cfg.OUTPUT_DIR).joinpath("training_images.txt")
        
            with open(train_image_output_path, mode="w") as f:
                for path in train_image_paths:
                    f.write(f"{path}\n")
    
    
    val_output_dir = None
    if val is not None:
        val = clean_input_paths(val)
        if not all((missing := path).exists() for path in val):
            raise FileNotFoundError(f"Validation File/Folder not found: {missing} does not exist")
        
        val_output_dir = output_dir.joinpath('val')
        process.set_input_paths(val)
        process.set_output_dir(val_output_dir)
        val_image_paths = get_file_paths(val, process.image_formats)
        process.run()
        
        if save_image_locations:
            #Saving the images used to a txt file
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            val_image_output_path = Path(cfg.OUTPUT_DIR).joinpath("validation_images.txt")
                
            with open(val_image_output_path, mode="w") as f:
                for path in val_image_paths:
                    f.write(f"{path}\n")
    
    dataset.register_datasets(train_output_dir, 
                              val_output_dir, 
                              train_name="train", 
                              val_name="val")

class Trainer(DefaultTrainer):
    """
    Trainer class
    """
    def __init__(self, cfg):
        TrainerBase.__init__(self)
        
        # logger = logging.getLogger("detectron2")
        # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        
        checkpoint_save_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        
        self.checkpointer = DetectionCheckpointer(
            model,
            checkpoint_save_dir,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        
        
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
        
        self.register_hooks(self.build_hooks() +
                            [miou_checkpointer, 
                             fwiou_checkpointer, 
                             macc_checkpointer, 
                             pacc_checkpointer])

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        sem_seg_output_dir = os.path.join(cfg.OUTPUT_DIR, "semantic_segmentation")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # TODO Other Evaluator types
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
        if cfg.MODEL.META_ARCHITECTURE in ["SemanticSegmentor", "MaskFormer", "PanopticFPN"]:
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
        if cfg.MODEL.META_ARCHITECTURE in ["SemanticSegmentor", "MaskFormer", "PanopticFPN"]:
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
    setup_logging(cfg, args)
    setup_seed(cfg)
    setup_saving(cfg)

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
    
    
