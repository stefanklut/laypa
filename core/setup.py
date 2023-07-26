import logging
import os
import subprocess
from typing import Optional, Sequence
from datetime import datetime
from pathlib import Path
import uuid

import torch

from detectron2.utils import comm
from detectron2.config import CfgNode, LazyConfig
from detectron2.utils.env import seed_all_rng
from detectron2.engine.defaults import _highlight

from configs.defaults import _C as _C_default
from configs.extra_defaults import _C as _C_extra

from utils.logging_utils import get_logger_name, setup_logger
from utils.path_utils import unique_path

# TODO Replace with LazyConfig
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
    
    _uuid = uuid.uuid4()
    cfg.LAYPA_UUID = str(_uuid)
    
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent).strip().decode()
    cfg.LAYPA_GIT_HASH = git_hash
    
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

def setup_logging(cfg=None, save_log=True):
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
        # REVIEW What to do with this, NLL is not deterministic. Only an issue during training
        # torch.use_deterministic_algorithms(True)