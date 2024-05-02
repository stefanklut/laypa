import argparse
import logging

from detectron2.evaluation import SemSegEvaluator

from core.preprocess import preprocess_datasets
from core.setup import setup_cfg, setup_logging, setup_saving, setup_seed
from core.trainer import Trainer
from utils.logging_utils import get_logger_name
from utils.tempdir import OptionalTemporaryDirectory


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation of model compared to ground truth")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", help="optional args to change", action="extend", default=[])

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    parser.add_argument("--sorted", action="store_true", help="Sorted iteration")
    parser.add_argument("--save", nargs="?", const="all", default=None, help="Save images instead of displaying")

    args = parser.parse_args()

    return args


def main(args):

    cfg = setup_cfg(args)
    setup_logging(cfg)
    setup_seed(cfg)
    setup_saving(cfg)

    logger = logging.getLogger(get_logger_name())

    # Temp dir for preprocessing in case no temporary dir was specified
    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not (args.keep_tmp_dir)) as tmp_dir:
        preprocess_datasets(cfg, None, args.input, tmp_dir)

        trainer = Trainer(cfg, validation=True)

        if not cfg.MODEL.WEIGHTS:
            logger.warning("No weights found in config or command line (MODEL.WEIGHTS), The model will be initialized randomly")
        trainer.resume_or_load(resume=False)

        trainer.validate()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
