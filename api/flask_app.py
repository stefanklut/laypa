import logging
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import torch
from flask import Flask, Response, abort, jsonify, request
from prometheus_client import Counter, Gauge, Summary, generate_latest

from utils.dict_utils import FIFOdict

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))  # noqa: E402
from data.mapper import AugInput
from inference import Predictor
from page_xml.output_page_xml import OutputPageXML
from page_xml.xml_regions import XMLRegions
from train import setup_cfg, setup_logging
from utils.image_utils import load_image_array_from_bytes
from utils.logging_utils import get_logger_name

# Reading environment files
try:
    max_queue_size_string: str = os.environ["LAYPA_MAX_QUEUE_SIZE"]
    model_base_path_string: str = os.environ["LAYPA_MODEL_BASE_PATH"]
    output_base_path_string: str = os.environ["LAYPA_OUTPUT_BASE_PATH"]
except KeyError as error:
    raise KeyError(f"Missing Laypa Environment variable: {error.args[0]}")

# Convert
max_queue_size = int(max_queue_size_string)
model_base_path = Path(model_base_path_string)
output_base_path = Path(output_base_path_string)

# Checks if ENV variable exist
if not model_base_path.is_dir():
    raise FileNotFoundError(f"LAYPA_MODEL_BASE_PATH: {model_base_path} is not found in the current filesystem")
if not output_base_path.is_dir():
    raise FileNotFoundError(f"LAYPA_OUTPUT_BASE_PATH: {output_base_path} is not found in the current filesystem")

# Capture logging
setup_logging()
logger = logging.getLogger(get_logger_name())

app = Flask(__name__)

predictor = None
gen_page = None


@dataclass
class DummyArgs:
    """
    Args to be used instead of the argparse.Namespace
    """

    config: str = "config.yaml"
    output: str = str(output_base_path)
    opts: list[str] = field(default_factory=list)


class PredictorGenPageWrapper:
    """
    Wrapper around the page generation code
    """

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.predictor: Optional[Predictor] = None
        self.gen_page: Optional[OutputPageXML] = None

    def setup_model(self, model_name: str, args: DummyArgs):
        """
        Create the model and post-processing code

        Args:
            model_name (str): Model name, used to determine what model to load from models present in base path
            args (DummyArgs): Dummy version of command line arguments, to set up config
        """
        # If model name matches current model name return without init
        if (
            model_name is not None
            and self.predictor is not None
            and self.gen_page is not None
            and model_name == self.model_name
        ):
            return

        self.model_name = model_name
        model_path = model_base_path.joinpath(self.model_name)
        config_path = model_path.joinpath("config.yaml")
        if not config_path.is_file():
            raise FileNotFoundError(f"config.yaml not found in {model_path}")
        weights_paths = list(model_path.glob("*.pth"))
        if len(weights_paths) < 1 or not weights_paths[0].is_file():
            raise FileNotFoundError(f"No valid .pth files found in {model_path}")
        if len(weights_paths) > 1:
            logger.warning(f"Found multiple .pth files. Using first {weights_paths[0]}")
        args.config = str(config_path)
        args.opts = ["TEST.WEIGHTS", str(weights_paths[0])]

        cfg = setup_cfg(args)
        xml_regions = XMLRegions(cfg)  # type: ignore
        self.gen_page = OutputPageXML(xml_regions=xml_regions, output_dir=None, cfg=cfg, whitelist={}, external_processing=True)

        self.predictor = Predictor(cfg=cfg)


args = DummyArgs()
predict_gen_page_wrapper = PredictorGenPageWrapper()

max_workers = 1
max_queue_size = max_workers + max_queue_size

# Run a separate thread on which the GPU runs and processes requests put in the queue
executor = ThreadPoolExecutor(max_workers=max_workers)

# Prometheus metrics to be returned
queue_size_gauge = Gauge("queue_size", "Size of worker queue").set_function(lambda: executor._work_queue.qsize())
images_processed_counter = Counter("images_processed", "Total number of images processed")
exception_predict_counter = Counter("exception_predict", "Exception thrown in predict() function")
process_time_summary = Summary("process_time", "Time taken to process an image")
queue_time_summary = Summary("queue_time", "Time taken to queue an image")
total_time_summary = Summary("total_time", "Total time taken to process an image")

ledger = FIFOdict(maxSize=1000000)


class ResponseInfo(dict):
    """
    Template for what fields are allowed in the response
    """

    allowed_fields = {
        "status_code",
        "identifier",
        "filename",
        "whitelist",
        "added_queue_position",
        "remaining_queue_size",
        "added_time",
        "model_name",
        "error_message",
        "status",
        "queue_time",
        "process_time",
        "total_time",
    }

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set the item in the dictionary, ignore if the value is None
        and ensure the key is in the allowed fields
        """
        if value is None:
            return
        if key not in self.allowed_fields:
            raise KeyError(f"Key {key} is not allowed in the response")
        super().__setitem__(key, value)


@dataclass
class Info:
    """
    Information about the current state of a request
    """

    _status: str = "loading"
    status_code: Optional[int] = None
    identifier: Optional[str] = None
    filename: Optional[str] = None
    whitelist: Optional[list[str]] = None
    added_queue_position: Optional[int] = None
    remaining_queue_size: Optional[int] = None
    model_name: Optional[str] = None
    error_message: Optional[str] = None
    start_time_queue: Optional[float] = None
    start_time_process: Optional[float] = None
    end_time_queue: Optional[float] = None
    end_time_process: Optional[float] = None

    @property
    def status(self) -> str:
        """
        Get the current status of the request

        Returns:
            str: Status of the request
        """

        return self._status

    @status.setter
    def status(self, value: str) -> None:
        """
        Set the status of the request

        Args:
            value (str): Status of the request
        """
        if value not in ["loading", "queued", "processing", "finished", "error"]:
            raise ValueError(f"Invalid status {value}. Must be one of ['loading', 'queued', 'processing', 'finished', 'error']")
        self._status = value

    @property
    def queue_time(self) -> Optional[float]:
        """
        Get the current queue time

        Returns:
            Optional[float]: Time in seconds
        """
        if self.start_time_queue is None:
            return None
        if not self.end_time_queue:
            end_time = time.time()
        else:
            end_time = self.end_time_queue
        queue_time = end_time - self.start_time_queue
        return queue_time

    @property
    def process_time(self) -> Optional[float]:
        """
        Get the current process time

        Returns:
            Optional[float]: Time in seconds
        """
        if self.start_time_process is None:
            return None
        if not self.end_time_process:
            end_time = time.time()
        else:
            end_time = self.end_time_process
        process_time = end_time - self.start_time_process
        return process_time

    @property
    def total_time(self) -> Optional[float]:
        """
        Get the current total time

        Returns:
            Optional[float]: Time in seconds
        """
        if self.start_time_queue is None:
            return None
        if not self.end_time_process:
            end_time = time.time()
        else:
            end_time = self.end_time_process
        total_time = end_time - self.start_time_queue
        return total_time

    @property
    def response_info(self) -> ResponseInfo:
        """
        Get the response info

        Returns:
            ResponseInfo: Response info
        """
        response_info = ResponseInfo()
        response_info["status_code"] = self.status_code or 500
        response_info["identifier"] = self.identifier
        response_info["filename"] = self.filename
        response_info["whitelist"] = self.whitelist
        response_info["added_queue_position"] = self.added_queue_position
        response_info["remaining_queue_size"] = self.remaining_queue_size
        response_info["added_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time_queue))
        response_info["model_name"] = self.model_name
        response_info["error_message"] = self.error_message
        response_info["status"] = self.status
        response_info["queue_time"] = self.queue_time
        response_info["process_time"] = self.process_time
        response_info["total_time"] = self.total_time
        return response_info


def safe_predict(data, device):
    """
    Attempt to predict on the specified data and device, handling CUDA out of memory errors gracefully.
    This function is a wrapper around the predictor to ensure that if a CUDA OOM error occurs, it falls back to CPU.

    Args:
        data: Data to predict on
        device: Device to predict on

    Returns:
        Prediction output

    Raises:
        TypeError: If the predictor is not initialized
    """

    try:
        if predict_gen_page_wrapper.predictor is None:
            raise TypeError("The predictor is not initialized. Ensure setup_model is called successfully.")
        return predict_gen_page_wrapper.predictor(data, device)
    except Exception as exception:
        # Catch CUDA out of memory errors
        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            isinstance(exception, RuntimeError) and "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(exception)
        ):
            logger.warning("CUDA OOM encountered, falling back to CPU.")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if predict_gen_page_wrapper.predictor is None:
                raise TypeError("The predictor is not initialized. Ensure setup_model is called successfully.")
            return predict_gen_page_wrapper.predictor(data, "cpu")


def predict_image(
    image: np.ndarray,
    dpi: Optional[int],
    image_path: Path,
    identifier: str,
    model_name: str,
    whitelist: list[str],
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

    Raises:
        TypeError: If the current GenPageXML is not initialized
        TypeError: If the current Predictor is not initialized
        ValueError: If the predictor did not return any outputs

    Returns:
        dict[str, Any]: Information about the processed image
    """
    info = ledger[identifier]
    info.status = "processing"
    current_time = time.time()
    info.start_time_process = current_time
    info.end_time_queue = current_time

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

        outputs = safe_predict(data, device=predict_gen_page_wrapper.predictor.cfg.MODEL.DEVICE)

        if outputs is None:
            raise ValueError("The predictor did not return any outputs")

        output_image = outputs[0]["sem_seg"]
        # output_image = torch.argmax(outputs[0]["sem_seg"], dim=-3).cpu().numpy()

        predict_gen_page_wrapper.gen_page.generate_single_page(
            output_image, output_path, old_height=outputs[1], old_width=outputs[2]
        )
        images_processed_counter.inc()
        return input_args
    except Exception as exception:
        # Catch CUDA out of memory errors
        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            isinstance(exception, RuntimeError) and "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(exception)
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # HACK remove traceback to prevent complete halt of program, not sure why this happens
            exception = exception.with_traceback(None)

        return input_args | {"exception": exception}


def abort_with_info(
    status_code: int,
    error_message: str,
    info: Optional[Info] = None,
):
    """
    Abort while still providing info about what went wrong

    Args:
        status_code (int): Error type code
        error_message (str): Message
        info (Optional[ResponseInfo], optional): Response info. Defaults to None.
    """
    if info is None:
        info = Info("error")

    info.status = "error"
    info.status_code = status_code
    info.error_message = error_message
    info.end_time_queue = time.time()

    response = jsonify(info.response_info)
    response.status_code = status_code
    abort(response)


def processing_done_callback(future: Future):
    """
    Log on exception

    Args:
        future (Future): Results from other thread
    """

    results = future.result()
    identifier = results["identifier"]
    info = ledger[identifier]
    info.end_time_process = time.time()
    if "exception" in results:
        logger.exception(results, exc_info=results["exception"])
        info.status = "error"
        info.error_message = str(results["exception"])
    else:
        info.status = "finished"

    process_time_summary.observe(info.process_time)
    queue_time_summary.observe(info.queue_time)
    total_time_summary.observe(info.total_time)


@app.route("/predict", methods=["POST"])
@exception_predict_counter.count_exceptions()
def predict() -> tuple[Response, int]:
    """
    Run the prediction on a submitted image

    Returns:
        Response: Submission response
    """
    if request.method != "POST":
        abort(405)

    info = Info()

    try:
        identifier = request.form["identifier"]
        info.identifier = identifier
    except KeyError as error:
        abort_with_info(400, "Missing identifier in form", info)

    # Add the identifier to the ledger
    ledger[identifier] = info

    try:
        model_name = request.form["model"]
        info.model_name = model_name
    except KeyError as error:
        abort_with_info(400, "Missing model in form", info)

    try:
        whitelist = request.form.getlist("whitelist")
        info.whitelist = whitelist
    except KeyError as error:
        abort_with_info(400, "Missing whitelist in form", info)

    try:
        post_file = request.files["image"]
    except KeyError as error:
        abort_with_info(400, "Missing image in form", info)

    if (image_name := post_file.filename) is not None:
        image_name = Path(image_name)
        info.filename = str(image_name)
    else:
        abort_with_info(400, "Missing filename", info)

    # TODO Maybe make slightly more stable/predicable, https://docs.python.org/3/library/threading.html#threading.Semaphore https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
    queue_size = executor._work_queue.qsize()
    info.added_queue_position = queue_size
    info.remaining_queue_size = max_queue_size - queue_size
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", info)

    img_bytes = post_file.read()
    data = load_image_array_from_bytes(img_bytes, image_path=image_name)

    if data is None:
        abort_with_info(500, "Image could not be loaded correctly", info)

    info.status = "queued"
    info.start_time_queue = time.time()

    future = executor.submit(predict_image, data["image"], data["dpi"], image_name, identifier, model_name, whitelist)  # type: ignore
    future.add_done_callback(processing_done_callback)

    info.status_code = 202
    # Return response and status code
    response = info.response_info
    response.pop("queue_time", None)
    response.pop("process_time", None)
    response.pop("total_time", None)
    return jsonify(response), 202


@app.route("/prometheus", methods=["GET"])
def metrics() -> bytes:
    """
    Return the Prometheus metrics for the running flask application

    Returns:
        bytes: Encoded string with the information
    """
    if request.method != "GET":
        abort(405)
    return generate_latest()


@app.route("/status_info", methods=["GET"])
def status_info() -> tuple[Response, int]:
    """
    Return the status of the current model

    Returns:
        Response: Status of the current model
    """
    if request.method != "GET":
        abort(405)

    info = Info()

    try:
        identifier = request.form["identifier"]
    except KeyError as error:
        abort_with_info(400, "Missing identifier in form", info)

    try:
        info = ledger[identifier]
    except KeyError as error:
        abort_with_info(400, "Identifier not found in ledger", info)

    return jsonify(info.response_info), 200


@app.route("/health", methods=["GET"])
def health_check() -> tuple[str, int]:
    """
    Health check endpoint for Kubernetes checks

    Returns:
        tuple[str, int]: Response and status code
    """
    return "OK", 200


if __name__ == "__main__":
    app.run()
