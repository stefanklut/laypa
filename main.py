# DEPRECATED This file has been replaced by the train.py file

import logging

from core.setup import setup_cfg, setup_logging
from train import get_arguments, main
from utils.logging_utils import get_logger_name

if __name__ == "__main__":
    args = get_arguments()
    cfg = setup_cfg(args)
    setup_logging(cfg, save_log=False)
    logger = logging.getLogger(get_logger_name())
    logger.warning(
        (
            "DeprecationWarning: The use of the main.py file is deprecated. "
            "Replace it with the train.py file in any scrips or commands you are using."
        )
    )

    main(args)
