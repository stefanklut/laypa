import argparse
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import datasets.dataset as dataset
import matplotlib.pyplot as plt
import cv2
import numpy as np
from main import setup_cfg
import torch.nn.functional as F
import torch

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")
    
    detectron2_args = parser.add_argument_group("detectron2")
    
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])
    
    other_args = parser.add_argument_group("other")
    other_args.add_argument("-t", "--train", help="Train input folder", type=str)
    other_args.add_argument("-v", "--val", help="Validation input folder", type=str)
    
    args = parser.parse_args()
    
    return args

class Predictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.TEST.WEIGHTS)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    def __call__(self, original_image):
        return super().__call__(original_image)

def main(args):
    cfg = setup_cfg(args)
    
    dataset.register(train=args.train,
                     val=args.val)
    
    predictor = Predictor(cfg=cfg)
    
    train_loader = dataset.dataset_dict_loader(args.train)
    val_loader = dataset.dataset_dict_loader(args.val)
    
    
    metadata = MetadataCatalog.get("pagexml_train")
    
    for inputs in np.random.choice(val_loader, 3):
        im = cv2.imread(inputs["file_name"])
        gt = cv2.imread(inputs["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
        outputs = predictor(im)
        outputs["sem_seg"] = torch.argmax(outputs["sem_seg"], dim=-3)
        print(inputs["file_name"])
        vis_im = Visualizer(im[:, :, ::-1].copy(),
                    metadata=metadata, 
                    scale=1
        )
        vis_im_gt = Visualizer(im[:, :, ::-1].copy(),
                    metadata=metadata, 
                    scale=1
        )
        vis_im = vis_im.draw_sem_seg(outputs["sem_seg"].to("cpu"))
        vis_im_gt = vis_im_gt.draw_sem_seg(gt)
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(vis_im.get_image()[:, :, ::-1])
        ax[0].axis('off')
        ax[1].imshow(vis_im_gt.get_image()[:, :, ::-1])
        ax[1].axis('off')
        # f.title(inputs["file_name"])
        plt.show()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
