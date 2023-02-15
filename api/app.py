from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.generate_pageXML import GenPageXML
from main import setup_cfg
from run import Predictor

app = Flask(__name__)

@dataclass
class DummyArgs():
    config: str = "configs/segmentation/baseline/baseline_dataset_imagenet.yaml"
    output: str = "./api/output/"
    opts: list[str] = field(default_factory=list)

args = DummyArgs()

cfg = setup_cfg(args, save_config=False)

#FIXME generates an empty output folder each time this is called
gen_page = GenPageXML(output_dir=args.output,
                      mode=cfg.MODEL.MODE,
                      line_width=cfg.PREPROCESS.BASELINE.LINE_WIDTH,
                      line_color=cfg.PREPROCESS.BASELINE.LINE_COLOR,
                      regions=cfg.PREPROCESS.REGION.REGIONS,
                      merge_regions=cfg.PREPROCESS.REGION.MERGE_REGIONS,
                      region_type=cfg.PREPROCESS.REGION.REGION_TYPE)

predictor = Predictor(cfg=cfg)

def load_image(img_bytes):
    # NOTE Reading images with cv2 removes the color profile, 
    # so saving the image results in different colors when viewing the image.
    # This should not affect the pixel values in the loaded image.
    bytes_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = file.read()
        image = load_image(img_bytes)
        outputs = predictor(image)
        output_image = torch.argmax(outputs["sem_seg"], dim=-3).cpu().numpy()
        return jsonify({"success": True})


if __name__ == '__main__':
    app.run()