import argparse
from typing import Optional
import torch
import detectron2
from detectron2.model_zoo import get
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.config import CfgNode

# TODO Replace with LazyConfig

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")
    
    detectron2_args = parser.add_argument_group("detectron2")
    
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts",nargs=argparse.REMAINDER, help="optional args to change")
    
    other_args = parser.add_argument_group("other")
    other_args.add_argument("--data_root", help="Data root")
    other_args.add_argument("--img_list", help="List with location of images")
    other_args.add_argument("--label_list", help="List with location of labels")
    other_args.add_argument("--out_size_list", help="List with sizes of images")
    
    
    args = parser.parse_args()
    
    return args
    

def setup_cfg(args, cfg: Optional[CfgNode]=None) -> CfgNode:
    if cfg is None:
        cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    
    return cfg

def main():
    
if __name__ == "__main__":
    main()