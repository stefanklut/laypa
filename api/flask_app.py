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
from prometheus_client import Counter, Gauge, generate_latest

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from main import setup_cfg, setup_logging
from page_xml.output_pageXML import OutputPageXML
from page_xml.xml_regions import XMLRegions
from run import Predictor
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
        xml_regions = XMLRegions(
            mode=cfg.MODEL.MODE,
            line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
            regions=cfg.PREPROCESS.REGION.REGIONS,
            merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
            region_type=cfg.PREPROCESS.REGION.REGION_TYPE,
        )
        self.gen_page = OutputPageXML(xml_regions=xml_regions, output_dir=None, cfg=cfg, whitelist={})

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


def predict_image(
    image: np.ndarray | torch.Tensor,
    image_path: Path,
    identifier: str,
    model_name: str,
    whitelist: list[str],
) -> dict[str, Any]:
    """
    Run the prediction for the given image

    Args:
        image (np.ndarray): Image array send to model prediction
        image_path (Path): Image name used to save image to location
        identifier (str): Unique identifier

    Raises:
        TypeError: The current GenPageXML in not initialized
        TypeError: The current Predictor is not initialized

    Returns:
        dict[str, Any]: Information about what image was processed
    """
    input_args = locals()
    try:
        predict_gen_page_wrapper.setup_model(args=args, model_name=model_name)

        output_path = output_base_path.joinpath(identifier, image_path)
        if predict_gen_page_wrapper.gen_page is None:
            raise TypeError("The current GenPageXML in not initialized")
        if predict_gen_page_wrapper.predictor is None:
            raise TypeError("The current Predictor is not initialized")

        predict_gen_page_wrapper.gen_page.set_output_dir(output_path.parent)
        predict_gen_page_wrapper.gen_page.set_whitelist(whitelist)
        if not output_path.parent.is_dir():
            output_path.parent.mkdir()

        outputs = predict_gen_page_wrapper.predictor(image)

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


class ResponseInfo(TypedDict, total=False):
    """
    Template for what fields are allowed in the response
    """

    status_code: int
    identifier: str
    filename: str
    whitelist: list[str]
    added_queue_position: int
    remaining_queue_size: int
    added_time: str
    model_name: str
    error_message: str


def abort_with_info(
    status_code: int,
    error_message: str,
    info: Optional[ResponseInfo] = None,
):
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
    response = jsonify(info)
    response.status_code = status_code
    abort(response)


def check_exception_callback(future: Future):
    """
    Log on exception

    Args:
        future (Future): Results from other thread
    """
    results = future.result()
    if "exception" in results:
        logger.exception(results, exc_info=results["exception"])


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

    response_info = ResponseInfo(status_code=500)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    response_info["added_time"] = current_time

    try:
        identifier = request.form["identifier"]
        response_info["identifier"] = identifier
    except KeyError as error:
        abort_with_info(400, "Missing identifier in form", response_info)

    try:
        model_name = request.form["model"]
        response_info["model_name"] = model_name
    except KeyError as error:
        abort_with_info(400, "Missing model in form", response_info)

    try:
        whitelist = request.form.getlist("whitelist")
        response_info["whitelist"] = whitelist
    except KeyError as error:
        abort_with_info(400, "Missing whitelist in form", response_info)

    try:
        post_file = request.files["image"]
    except KeyError as error:
        abort_with_info(400, "Missing image in form", response_info)

    if (image_name := post_file.filename) is not None:
        image_name = Path(image_name)
        response_info["filename"] = str(image_name)
    else:
        abort_with_info(400, "Missing filename", response_info)

    # TODO Maybe make slightly more stable/predicable, https://docs.python.org/3/library/threading.html#threading.Semaphore https://gist.github.com/frankcleary/f97fe244ef54cd75278e521ea52a697a
    queue_size = executor._work_queue.qsize()
    response_info["added_queue_position"] = queue_size
    response_info["remaining_queue_size"] = max_queue_size - queue_size
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", response_info)

    img_bytes = post_file.read()
    image = load_image_array_from_bytes(img_bytes, image_path=image_name)

    if image is None:
        abort_with_info(500, "Image could not be loaded correctly", response_info)

    future = executor.submit(predict_image, image, image_name, identifier, model_name, whitelist)
    future.add_done_callback(check_exception_callback)

    response_info["status_code"] = 202
    # Return response and status code
    return jsonify(response_info), 202


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
