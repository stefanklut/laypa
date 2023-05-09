import argparse
from functools import lru_cache
import logging
import random
import sys
from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt
import cv2
import numpy as np
from core.preprocess import preprocess_datasets
from core.setup import setup_cfg
import torch
from natsort import os_sorted
from utils.logging_utils import get_logger_name
from utils.tempdir import OptionalTemporaryDirectory
from run import Predictor


logger = logging.getLogger(get_logger_name())

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval of prediction of model using visualizer")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument(
        "-c", "--config", help="config file", required=True)
    detectron2_args.add_argument(
        "--opts", nargs="+", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-t", "--train", help="Train input folder/file",
                            nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-v", "--val", help="Validation input folder/file",
                            nargs="+", action="extend", type=str, default=None)
    
    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument(
        "--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument(
        "--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")
    
    parser.add_argument("--eval_path", type=str, help="Save location for eval")
    parser.add_argument("--sorted", action="store_true", help="Sorted iteration")
    parser.add_argument("--save", action="store_true", help="Save images instead of displaying")

    args = parser.parse_args()

    return args

_keypress_result = None 
def keypress(event):
    global _keypress_result
    # print('press', event.key)
    if event.key in ['q', 'escape']:
        sys.exit()
    if event.key in [' ', 'right']:
        _keypress_result = "forward"
        return
    if event.key in ['backspace', 'left']:
        _keypress_result = "back"
        return
    if event.key in ['e', 'delete']:
        _keypress_result = "delete"
        return
    if event.key in ['w']:
        _keypress_result = "bad"
        return
    
def on_close(event):
    sys.exit()

def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    if args.save and not args.eval_path:
        raise ValueError("Cannot run saving when there is not save location given (--eval_path)")

    # Setup config
    cfg = setup_cfg(args)

    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not(args.keep_tmp_dir)) as tmp_dir:
        
        preprocess_datasets(cfg, args.train, args.val, tmp_dir, save_image_locations=False)
        predictor = Predictor(cfg=cfg)
        
        # train_loader = DatasetCatalog.get("train")
        val_loader = DatasetCatalog.get("val")
        metadata = MetadataCatalog.get("val")
        # print(metadata)
        
        
        @lru_cache(maxsize=10)
        def load_image(filename):
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            return image

        @lru_cache(maxsize=10)
        def load_sem_seg(filename):
            sem_seg_gt = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return sem_seg_gt

        @lru_cache(maxsize=10)
        def create_gt_visualization(image_filename, sem_seg_filename):
            image = load_image(image_filename)
            sem_seg_gt = load_sem_seg(sem_seg_filename)
            vis_im_gt = Visualizer(image[..., ::-1].copy(),
                                    metadata=metadata,
                                    scale=1
                                    )
            vis_im_gt = vis_im_gt.draw_sem_seg(sem_seg_gt, alpha=0.4)
            return vis_im_gt.get_image()

        @lru_cache(maxsize=10)
        def create_pred_visualization(image_filename):
            image = load_image(image_filename)
            logger.info(f"Predict: {image_filename}")
            outputs = predictor(image)
            pred = torch.argmax(outputs[0]["sem_seg"], dim=-3).to("cpu")
            # outputs["panoptic_seg"] = (outputs["panoptic_seg"][0].to("cpu"), 
            #                            outputs["panoptic_seg"][1])
            vis_im = Visualizer(image[..., ::-1].copy(),
                                metadata=metadata,
                                scale=1
                                )
            
            vis_im = vis_im.draw_sem_seg(pred, alpha=0.4)
            return vis_im.get_image()
        
        fig, axes = plt.subplots(1, 2)
        fig.tight_layout()
        fig.canvas.mpl_connect('key_press_event', keypress)
        fig.canvas.mpl_connect('close_event', on_close)
        axes[0].axis('off')
        axes[1].axis('off')
        if not args.save:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
        
        # for i, inputs in enumerate(np.random.choice(val_loader, 3)):
        if args.sorted:
            loader = os_sorted(val_loader, key=lambda x: x["file_name"])
        else:
            loader = val_loader
            random.shuffle(val_loader)
            
        bad_results = np.zeros(len(loader), dtype=bool)
        delete_results = np.zeros(len(loader), dtype=bool)
        
        i = 0
        while 0<=i<len(loader):
            inputs = loader[i]
            
            vis_gt = create_gt_visualization(inputs["file_name"], inputs["sem_seg_file_name"])
            vis_pred = create_pred_visualization(inputs["file_name"])
            
            # pano_gt = torch.IntTensor(rgb2id(cv2.imread(inputs["pan_seg_file_name"], cv2.IMREAD_COLOR)))
            # print(inputs["segments_info"])
            
            # vis_im = vis_im.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
            # vis_im_gt = vis_im_gt.draw_panoptic_seg(pano_gt, [item | {"isthing": True} for item in inputs["segments_info"]])
            if not args.save:
                fig_manager.window.setWindowTitle(inputs["file_name"])
            
            # HACK Just remove the previous axes, I can't find how to resize the image otherwise
            axes[0].clear()
            axes[1].clear()
            axes[0].axis('off')
            axes[1].axis('off')

            axes[0].imshow(vis_pred)
            axes[1].imshow(vis_gt)
            
            if args.save:
                output_dir = Path(args.eval_path)
                if not output_dir.is_dir():
                    logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
                    output_dir.mkdir(parents=True)
                save_path = output_dir.joinpath(Path(inputs["file_name"]).stem + ".png")
                
                # Save to 4K res
                
                fig.set_size_inches(16, 9)
                fig.savefig(str(save_path), dpi=240)
                i += 1
                continue
            
            if delete_results[i]:
                fig.suptitle('Delete')
            elif bad_results[i]:
                fig.suptitle('Bad')
            else:
                fig.suptitle("")
            # f.title(inputs["file_name"])
            global _keypress_result
            _keypress_result = None
            fig.canvas.draw()
            while _keypress_result is None:
                plt.waitforbuttonpress()
            if _keypress_result == "delete":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                delete_results[i] = not delete_results[i]
                bad_results[i] = False
            elif _keypress_result == "bad":
                # print(i+1, f"{inputs['original_file_name']}: BAD")
                bad_results[i] = not bad_results[i]
                delete_results[i] = False
            elif _keypress_result == "forward":
                # print(i+1, f"{inputs['original_file_name']}")
                i += 1
            elif _keypress_result == "back":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                i -= 1
                
    if args.eval_path and (delete_results.any() or bad_results.any()):
        output_dir = Path(args.eval_path)
        if not output_dir.is_dir():
            logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)
        if delete_results.any():
            eval_path_delete = output_dir.joinpath("delete.txt")
            with eval_path_delete.open(mode="w") as f:
                for i in delete_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")
        if bad_results.any():
            eval_path_bad = output_dir.joinpath("bad.txt")
            with eval_path_bad.open(mode="w") as f:
                for i in bad_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")
            
        remaining_results = np.logical_not(np.logical_or(bad_results, delete_results))
        if remaining_results.any():
            eval_path_remaining = output_dir.joinpath("correct.txt")
            with eval_path_remaining.open(mode="w") as f:
                for i in remaining_results.nonzero()[0]:
                    path = Path(loader[i]["original_file_name"])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")

if __name__ == "__main__":
    args = get_arguments()
    main(args)
