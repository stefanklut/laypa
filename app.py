import io
import json
import torch

import numpy as np
import cv2

from flask import Flask, jsonify, request
from page_xml.generate_pageXML import GenPageXML
from run import Predictor
from main import setup_cfg
from dataclasses import dataclass, field


app = Flask(__name__)

@dataclass
class DummyArgs():
    config: str = "configs/segmentation/baseline/baseline_dataset_imagenet.yaml"
    output: str = "test/"
    opts: list[str] = field(default_factory=list)

args = DummyArgs()

cfg = setup_cfg(args, save_config=False)

gen_page = GenPageXML(output_dir=args.output,
                      mode=cfg.MODEL.MODE,
                      line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
                      line_color=cfg.PREPROCESS.BASELINE.LINE_COLOR,
                      regions=cfg.PREPROCESS.REGION.REGIONS,
                      merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
                      region_type=cfg.PREPROCESS.REGION.REGION_TYPE)

predictor = Predictor(cfg=cfg)

def load_image(img_bytes):
    bytes_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        image = load_image(img_bytes)
        outputs = predictor(image)
        output_image = torch.argmax(outputs["sem_seg"], dim=-3).cpu().numpy()
        return jsonify({"success": True})


if __name__ == '__main__':
    app.run()