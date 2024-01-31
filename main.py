import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from core.preprocess import preprocess_datasets
from core.setup import setup_cfg, setup_logging, setup_saving, setup_seed

root_logger = logging.getLogger()

from detectron2.engine import launch
from detectron2.engine.defaults import _highlight
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info

import models
from core.trainer import Trainer
from utils.logging_utils import get_logger_name
from utils.tempdir import OptionalTemporaryDirectory

# torch.autograd.set_detect_anomaly(True)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", action="extend", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument(
        "-v", "--val", help="Validation input folder/file", nargs="+", action="extend", type=str, required=True
    )

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    # other_args.add_argument("--img_list", help="List with location of images")
    # other_args.add_argument("--label_list", help="List with location of labels")
    # other_args.add_argument("--out_size_list", help="List with sizes of images")

    # From detectron2.engine.defaults
    gpu_args = parser.add_argument_group("GPU launch")
    gpu_args.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    gpu_args.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    gpu_args.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

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


def setup_training(args: argparse.Namespace):
    """
    Setup and start training

    Args:
        args (argparse.Namespace): arguments used to load a config file, also used for overwriting values directly (--opts)

    Returns:
        OrderedDict|None: results, if evaluation is enabled. Otherwise None.
    """
    cfg = setup_cfg(args)
    setup_logging(cfg)
    setup_seed(cfg)
    setup_saving(cfg)

    logger = logging.getLogger(get_logger_name())

    rank = comm.get_rank()

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    if args is not None:
        logger.info("Command line arguments: " + str(args))
        if hasattr(args, "config") and args.config != "":
            with Path(args.config).open("r") as f:
                config_contents = f.read()
            logger.info(
                "Contents of args.config: {}:\n{}".format(
                    args.config,
                    _highlight(config_contents, args.config),
                )
            )

    # Temp dir for preprocessing in case no temporary dir was specified
    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not (args.keep_tmp_dir)) as tmp_dir:
        preprocess_datasets(cfg, args.train, args.val, tmp_dir)

        trainer = Trainer(cfg=cfg)
        if not cfg.TRAIN.WEIGHTS:
            if cfg.MODEL.RESUME and trainer.checkpointer.has_checkpoint():
                raise FileNotFoundError(f"No checkpoint found in {cfg.OUTPUT_DIR}")
            trainer.resume_or_load(resume=cfg.MODEL.RESUME)
        else:
            trainer.checkpointer.load(cfg.TRAIN.WEIGHTS)
            trainer.start_iter = trainer.iter + 1

        results = trainer.train()

    return results


def main(args: argparse.Namespace) -> None:
    assert (
        args.num_gpus <= torch.cuda.device_count()
    ), f"Less GPUs found ({torch.cuda.device_count()}) than specified ({args.num_gpus})"

    launch(
        setup_training,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
