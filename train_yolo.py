import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

from core.setup import setup_cfg, setup_logging, setup_saving, setup_seed
from data.preprocess_yolo import PreprocessYOLO

root_logger = logging.getLogger()

from detectron2.engine import launch
from detectron2.engine.defaults import _highlight
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from ultralytics import YOLO

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

    yolo_args = parser.add_argument_group("YOLO")
    yolo_args.add_argument("--yolo", help="yolo model", type=str, default="yolo11n.pt")

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

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

    if cfg.INPUT.ON_GPU:
        torch.multiprocessing.set_start_method("spawn", force=True)

    # Find the yolo type
    model = YOLO(args.yolo)
    yolo_task = model.task
    logger.info(f"Yolo task: {yolo_task}")

    # Temp dir for preprocessing in case no temporary dir was specified
    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not (args.keep_tmp_dir)) as tmp_dir:
        process = PreprocessYOLO(cfg, yolo_task=yolo_task)  # type: ignore

        tmp_dir = Path(tmp_dir)

        train_output_dir = tmp_dir.joinpath("train")
        process.set_input_paths(args.train)
        process.set_output_dir(train_output_dir)
        process.run()

        val_output_dir = tmp_dir.joinpath("val")
        process.set_input_paths(args.val)
        process.set_output_dir(val_output_dir)
        process.run()

        # HACK invert the paths and move the images to the correct folder
        image_train_path = train_output_dir.joinpath("image")
        labels_train_path = train_output_dir.joinpath("labels")
        image_val_path = val_output_dir.joinpath("image")
        labels_val_path = val_output_dir.joinpath("labels")

        images_path = tmp_dir.joinpath("images")
        images_path.mkdir(parents=True, exist_ok=True)
        image_train_path.rename(images_path.joinpath("train"))
        image_val_path.rename(images_path.joinpath("val"))

        labels_path = tmp_dir.joinpath("labels")
        labels_path.mkdir(parents=True, exist_ok=True)
        labels_train_path.rename(labels_path.joinpath("train"))
        labels_val_path.rename(labels_path.joinpath("val"))

        shutil.rmtree(train_output_dir)
        shutil.rmtree(val_output_dir)

        names = {i: name for i, name in enumerate(process.xml_regions.regions[1:])}  # 1: to exclude background class

        yolo_output_path = tmp_dir.joinpath("yolo.yaml")
        yolo_output = {
            "path": str(tmp_dir),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }

        with yolo_output_path.open("w") as f:
            yaml.dump(yolo_output, f)

        model.train(
            data=str(yolo_output_path),
            epochs=100,
            batch=cfg.SOLVER.IMS_PER_BATCH,
            imgsz=640,
            device=cfg.MODEL.DEVICE,
            workers=cfg.DATALOADER.NUM_WORKERS,
        )


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
