# Imports

# > Standard Libraries
from pathlib import Path
from typing import Any, Optional
import time
import logging

# > External Libraries
import numpy as np
import torch
from flask import current_app, Request

# > Project Libraries
from api.models.response_info import ResponseInfo
from api.services.utils import abort_with_info, check_exception_callback
from api.services.image_processing import safe_predict
from api.routes.utils import extract_request_fields
from data.mapper import AugInput
from utils.image_utils import load_image_array_from_bytes
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())


def process_prediction(request: Request) -> ResponseInfo:
    """
    Start the prediction process for the given image by submitting it to the
    executor

    Args:
        request (Request): Request object from the API

    Returns:
        ResponseInfo: Information about the processed image
    """

    executor = current_app.executor
    args = current_app.args
    output_base_path = current_app.output_base_path
    predict_gen_page_wrapper = current_app.predict_gen_page_wrapper
    images_processed_counter = current_app.images_processed_counter
    max_queue_size = current_app.max_queue_size

    response_info = ResponseInfo(status_code=500)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    response_info["added_time"] = current_time

    post_file, image_name, identifier, model_name, whitelist = extract_request_fields(request, response_info)

    queue_size = executor._work_queue.qsize()
    response_info["added_queue_position"] = queue_size
    response_info["remaining_queue_size"] = max_queue_size - queue_size
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", response_info)

    img_bytes = post_file.read()
    data = load_image_array_from_bytes(img_bytes, image_path=image_name)

    if data is None:
        abort_with_info(500, "Image could not be loaded correctly", response_info)

    future = executor.submit(
        predict_image,
        data["image"],
        data["dpi"],
        image_name,
        identifier,
        model_name,
        whitelist,
        predict_gen_page_wrapper,
        output_base_path,
        images_processed_counter,
        args,
    )
    future.add_done_callback(check_exception_callback)
    response_info["status_code"] = 202

    logger.info(f"Image {image_name} added to queue")
    return response_info


def predict_image(
    image: np.ndarray,
    dpi: Optional[int],
    image_path: Path,
    identifier: str,
    model_name: str,
    whitelist: list[str],
    predict_gen_page_wrapper,
    output_base_path: Path,
    images_processed_counter,
    args: Any,
) -> dict[str, Any]:
    """
    Run the prediction for the given image

    Args:
        image (np.ndarray): Image array send to model prediction
        dpi (Optional[int]): DPI (dots per inch) of the image
        image_path (Path): Path to the image file
        identifier (str): Unique identifier for the image
        model_name (str): Name of the model to use for prediction
        whitelist (list[str]): List of characters to whitelist during prediction
        predict_gen_page_wrapper: Wrapper for the Predictor and GenPageXML
        output_base_path (Path): Base path for the output
        images_processed_counter: Counter for the number of images processed
        args (Any): Arguments for the model setup

    Raises:
        TypeError: If the current GenPageXML is not initialized
        TypeError: If the current Predictor is not initialized

    Returns:
        dict[str, Any]: Information about the processed image
    """
    input_args = locals()
    try:
        predict_gen_page_wrapper.setup_model(args=args, model_name=model_name)

        output_path = output_base_path.joinpath(identifier, image_path)
        if predict_gen_page_wrapper.gen_page is None:
            raise TypeError("The current GenPageXML is not initialized")
        if predict_gen_page_wrapper.predictor is None:
            raise TypeError("The current Predictor is not initialized")

        predict_gen_page_wrapper.gen_page.set_output_dir(output_path.parent)
        predict_gen_page_wrapper.gen_page.set_whitelist(whitelist)
        if not output_path.parent.is_dir():
            output_path.parent.mkdir()

        data = AugInput(
            image=image,
            dpi=dpi,
            auto_dpi=predict_gen_page_wrapper.predictor.cfg.INPUT.DPI.AUTO_DETECT_TEST,
            default_dpi=predict_gen_page_wrapper.predictor.cfg.INPUT.DPI.DEFAULT_DPI_TEST,
            manual_dpi=predict_gen_page_wrapper.predictor.cfg.INPUT.DPI.MANUAL_DPI_TEST,
        )

        outputs = safe_predict(
            data, device=predict_gen_page_wrapper.predictor.cfg.MODEL.DEVICE, predict_gen_page_wrapper=predict_gen_page_wrapper
        )

        output_image = outputs[0]["sem_seg"]

        predict_gen_page_wrapper.gen_page.generate_single_page(
            output_image, output_path, old_height=outputs[1], old_width=outputs[2]
        )
        images_processed_counter.inc()

        logger.info(f"Prediction complete for {image_path}")
        return input_args
    except Exception as exception:
        # Catch CUDA out of memory errors
        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            isinstance(exception, RuntimeError) and "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(exception)
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # HACK remove traceback to prevent complete halt of program, not
            # sure why this happens
            exception = exception.with_traceback(None)

        return input_args | {"exception": exception}
