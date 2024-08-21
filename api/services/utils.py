# Imports

# > Standard Libraries
import logging
from concurrent.futures import Future
from typing import Optional

# > External Libraries
from flask import jsonify, abort

# > Project Libraries
from api.models.response_info import ResponseInfo


def abort_with_info(status_code: int, error_message: str, info: Optional[ResponseInfo] = None):
    """
    Abort while still providing info about what went wrong

    Args:
        status_code (int): Error type code
        error_message (str): Message
        info (Optional[ResponseInfo], optional): Response info. Defaults to None.
    """
    if info is None:
        info = ResponseInfo(status_code=status_code)  # type: ignore
    info["error_message"] = error_message
    info["status_code"] = status_code
    response = jsonify(info)
    response.status_code = status_code
    abort(response)


def check_exception_callback(future: Future):
    """
    Log on exception

    Args:
        future (Future): Results from other thread
    """
    logger = logging.getLogger(__name__)
    results = future.result()
    if "exception" in results:
        logger.exception(results, exc_info=results["exception"])
