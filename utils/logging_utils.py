# Based on detectron2.utils.logger.py

import functools
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from detectron2.utils.logger import _cached_log_stream
from termcolor import colored


def get_logger_name():
    frame = sys._getframe(1)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logging_utils.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "laypa"
            else:
                mod_name = "laypa." + mod_name
            # return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
            return mod_name
        frame = frame.f_back


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record: logging.LogRecord) -> str:
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("DEBUG", "blue", attrs=["blink"])
        elif record.levelno == logging.INFO:
            return log
        else:
            return log
        return prefix + " " + log


class _PlainFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs) -> None:
        super(_PlainFormatter, self).__init__(*args, **kwargs)

    @staticmethod
    def remove_ansi(message: str) -> str:
        pattern = re.compile(r"\x1B\[\d+(;\d+){0,2}m")
        stripped = pattern.sub("", message)
        return stripped

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        message = self.remove_ansi(message)
        return message


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(output=None, distributed_rank=0, *, color=True, name="detectron2", abbrev_name=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = _PlainFormatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        output_path = Path(output)
        if output_path.suffix in [".txt" ".log"]:
            filename = output_path
        else:
            filename = output_path.joinpath("log.txt")
        if distributed_rank > 0:
            filename = Path(f"{filename}.rank{distributed_rank}")
        filename.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(str(filename)))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger
