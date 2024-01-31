import logging
import os
import subprocess
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
from detectron2.config import CfgNode, LazyConfig
from detectron2.engine.defaults import _highlight
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng

from configs.defaults import _C as _C_default
from configs.extra_defaults import _C as _C_extra
from utils.logging_utils import get_logger_name, setup_logger
from utils.path_utils import unique_path


def setup_logging(cfg: Optional[CfgNode] = None, save_log: bool = True) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        cfg (Optional[CfgNode]): Application configuration. Defaults to None.
        save_log (bool): Whether to save log files. Defaults to True.

    Returns:
        logging.Logger: Logger object.
    """
    rank = comm.get_rank()
    if save_log and cfg is not None:
        output_dir = cfg.OUTPUT_DIR
    else:
        output_dir = None
    root_logger = logging.getLogger()
    logging.getLogger("fvcore")
    logging.getLogger("detectron2")

    logger = setup_logger(output_dir, distributed_rank=rank, name="laypa")

    for item in root_logger.manager.loggerDict:
        if item.startswith("detectron2") or item.startswith("fvcore"):
            root_logger.manager.loggerDict[item] = logger

    return logger


def get_git_hash() -> str:
    version_path = Path("version_info")

    if version_path.is_file():
        with version_path.open(mode="r") as file:
            git_hash = file.read()
    else:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent).strip().decode()
    return git_hash


# TODO Replace with LazyConfig
def setup_cfg(args, cfg: Optional[CfgNode] = None) -> CfgNode:
    """
    Create the config used for training and evaluation.
    Loads from default configs and merges with a specific config file specified in the command line arguments.

    Args:
        args (argparse.Namespace): Command line arguments used to load a config file and overwrite values directly (--opts).
        cfg (Optional[CfgNode]): Possible overwrite of the default config. Defaults to None.

    Returns:
        CfgNode: Configuration for training and evaluation.
    """

    logger = logging.getLogger(get_logger_name())
    if cfg is None:
        cfg = _C_default

    cfg.defrost()

    # Merge with extra defaults, config file, and command line args
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

    cfg.LAYPA_GIT_HASH = get_git_hash()

    config_path = Path(args.config).resolve()
    cfg.CONFIG_PATH = str(config_path)

    if not hasattr(args, "train"):
        pass
    elif args.train is None:
        pass
    elif isinstance(args.train, Sequence):
        cfg.TRAINING_PATHS = [str(Path(path).resolve()) for path in args.train]
    else:
        cfg.TRAINING_PATHS = [str(Path(args.train).resolve())]

    if not hasattr(args, "val"):
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

    if cfg.MODEL.RESUME:
        cfg.OUTPUT_DIR = str(config_path.parent)

    # Overwrite device based on detected hardware
    if cfg.MODEL.DEVICE:
        # If CUDA device cannot be found, default to CPU
        if torch.cuda.device_count() == 0:
            cfg.MODEL.DEVICE = "cpu"
    else:
        # If not specified, use CUDA if possible, otherwise, use CPU
        if torch.cuda.device_count() > 0:
            cfg.MODEL.DEVICE = "cuda"
        else:
            cfg.MODEL.DEVICE = "cpu"

    # Deprecation warnings
    if cfg.PREPROCESS.RESIZE.USE:
        logger.warning(
            "DeprecationWarning PREPROCESS.RESIZE.USE is losing support; please switch to PREPROCESS.RESIZE.RESIZE_MODE"
        )
        cfg.PREPROCESS.RESIZE.RESIZE_MODE = "shortest_edge"

    if cfg.INPUT.SCALING:
        logger.warning(
            "DeprecationWarning INPUT.SCALING is losing support; please switch to INPUT.SCALING_TRAIN and INPUT.SCALING_TEST"
        )
        cfg.INPUT.SCALING_TRAIN = cfg.INPUT.SCALING

    if cfg.INPUT.SCALING_TEST <= 0.0:
        test_scaling = cfg.INPUT.SCALING_TRAIN * cfg.PREPROCESS.RESIZE.SCALING
        logger.warning(
            f"INPUT.SCALING_TEST is not set, inferring from INPUT.SCALING_TRAIN and PREPROCESS.RESIZE.SCALING to be {test_scaling}"
        )
        cfg.INPUT.SCALING_TEST = test_scaling

    cfg.freeze()
    return cfg


def setup_saving(cfg: CfgNode):
    """
    Set up saving configurations.

    Args:
        cfg (CfgNode): Configuration node.
    """
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
    """
    Set up the random seed for reproducibility.

    Args:
        cfg (CfgNode): Configuration node.
    """
    seed = cfg.SEED if cfg.SEED else -1
    rank = comm.get_rank()
    seed_all_rng(None if seed < 0 else seed + rank)
    # REVIEW What to do with this, NLL is not deterministic. Only an issue during training
    # torch.use_deterministic_algorithms(True)
