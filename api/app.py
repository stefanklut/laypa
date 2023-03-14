import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Optional, TypedDict

import cv2
import numpy as np
import torch
from flask import Flask, abort, jsonify, request

from prometheus_client import generate_latest, Counter, Gauge

from concurrent.futures import ThreadPoolExecutor

from utils.image_utils import load_image_from_bytes

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from main import setup_cfg
from page_xml.generate_pageXML import GenPageXML
from run import Predictor

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

    
app = Flask(__name__)

predictor = None
gen_page = None

@dataclass
class DummyArgs():
    config: str = "config.yaml"
    output: str = str(output_base_path)
    opts: list[str] = field(default_factory=list)


class PredictorGenPageWrapper():
    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.predictor: Optional[Predictor] = None
        self.gen_page: Optional[GenPageXML] = None

    def setup_model(self, model_name, args):
        if model_name == self.model_name:
            return
        
        self.model_name = model_name
        config_path = model_base_path.joinpath(self.model_name, "config.yaml")
        weights_path = next(model_base_path.joinpath(self.model_name).glob("*.pth"))
        
        args.config = str(config_path)
        args.opts = ["TEST.WEIGHTS", str(weights_path)]
        
        cfg = setup_cfg(args, save_config=False)

        self.gen_page = GenPageXML(mode=cfg.MODEL.MODE,
                                   output_dir=None,
                                   line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
                                   line_color=cfg.PREPROCESS.BASELINE.LINE_COLOR,
                                   regions=cfg.PREPROCESS.REGION.REGIONS,
                                   merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
                                   region_type=cfg.PREPROCESS.REGION.REGION_TYPE)

        self.predictor = Predictor(cfg=cfg)

args = DummyArgs()
predict_gen_page_wrapper = PredictorGenPageWrapper()

max_workers = 1
max_queue_size = max_workers + max_queue_size

executor = ThreadPoolExecutor(max_workers=max_workers)

queue_size_gauge = Gauge('queue_size', "Size of worker queue").set_function(lambda: executor._work_queue.qsize())
images_processed_counter = Counter('images_processed', "Total number of images processed")
exception_predict_counter = Counter('exception_predict', 'Exception thrown in predict() function')

def predict_image(image: np.ndarray, image_path: Path, identifier: str):
    output_path = output_base_path.joinpath(identifier, image_path)
    if predict_gen_page_wrapper.gen_page is None:
        raise ValueError
    if predict_gen_page_wrapper.predictor is None:
        raise ValueError
    
    predict_gen_page_wrapper.gen_page.set_output_dir(output_path.parent)
    if not output_path.parent.exists():
        output_path.parent.mkdir()
    
    outputs = predict_gen_page_wrapper.predictor(image)
    output_image = torch.argmax(outputs["sem_seg"], dim=-3).cpu().numpy()
    predict_gen_page_wrapper.gen_page.generate_single_page(output_image, output_path)
    images_processed_counter.inc()
    return True


class ResponseInfo(TypedDict, total=False):
    success: bool
    identifier: str
    filename: str
    added_queue_position: int
    added_time: str
    model_name: str
    error_message: str


def abort_with_info(status_code, error_message, info: Optional[ResponseInfo]=None):
    if info is None:
        info = ResponseInfo(success=False) # type: ignore
    info["error_message"] = error_message
    response = jsonify(info)
    response.status_code = status_code
    abort(response)

@app.route('/predict', methods=['POST'])
@exception_predict_counter.count_exceptions()
def predict():
    if request.method != 'POST':
        abort(400)
        
    response_info = ResponseInfo(success=False)
    
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
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
        post_file = request.files['image']
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
    if queue_size > max_queue_size:
        abort_with_info(429, "Exceeding queue size", response_info)
    
    predict_gen_page_wrapper.setup_model(args=args, model_name=model_name)
    
    img_bytes = post_file.read()
    image = load_image_from_bytes(img_bytes)
    
    if image is None:
        abort_with_info(400, "Corrupted image", response_info)
    
    future = executor.submit(predict_image, image, image_name, identifier)
    # TODO Add callbacks
    response_info["success"] = True
    return jsonify(response_info)


@app.route('/prometheus', methods=['GET'])
def metrics():
    return generate_latest()


if __name__ == '__main__':
    app.run()