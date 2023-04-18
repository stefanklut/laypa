import argparse
import sys
from pathlib import Path

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import matplotlib.pyplot as plt
import cv2
from main import preprocess_datasets, setup_cfg
import torch
from natsort import os_sorted
from utils.tempdir import OptionalTemporaryDirectory
from run import Predictor


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval of prediction of model using visualizer")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument(
        "-c", "--config", help="config file", required=True)
    detectron2_args.add_argument(
        "--opts", nargs=argparse.REMAINDER, help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-t", "--train", help="Train input folder", type=str, default=None)
    io_args.add_argument(
        "-v", "--val", help="Validation input folder", type=str, default=None)
    
    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument(
        "--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument(
        "--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")
    
    parser.add_argument("--eval_path", type=str, help="Save location for eval")

    args = parser.parse_args()

    return args

_keypress_result = None 
def keypress(event):
    global _keypress_result
    # print('press', event.key)
    if event.key == 'q':
        sys.exit()
    if event.key == ' ':
        _keypress_result = "continue"
        return
    if event.key == 'w':
        _keypress_result = "delete"
        return
    if event.key == 'e':
        _keypress_result = "bad"
        return

def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    # Setup config

    cfg = setup_cfg(args)
    
    bad_results = []
    delete_results = []
    remaining_results = []

    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not(args.keep_tmp_dir)) as tmp_dir:
        
        preprocess_datasets(cfg, args.train, args.val, tmp_dir, save_image_locations=False)
        predictor = Predictor(cfg=cfg)
        
        # train_loader = DatasetCatalog.get("train")
        val_loader = DatasetCatalog.get("val")
        metadata = MetadataCatalog.get("val")
        # print(metadata)
        
        f, ax = plt.subplots(1, 2)
        
        # for i, inputs in enumerate(np.random.choice(val_loader, 3)):
        for i, inputs in enumerate(os_sorted(val_loader, key=lambda x: x["file_name"])):
            im = cv2.imread(inputs["file_name"])
            sem_seg_gt = cv2.imread(inputs["sem_seg_file_name"], cv2.IMREAD_GRAYSCALE)
            # pano_gt = torch.IntTensor(rgb2id(cv2.imread(inputs["pan_seg_file_name"], cv2.IMREAD_COLOR)))
            # print(inputs["segments_info"])
            
            outputs = predictor(im)
            # print(outputs)
            outputs["sem_seg"] = torch.argmax(outputs["sem_seg"], dim=-3).to("cpu")
            # outputs["panoptic_seg"] = (outputs["panoptic_seg"][0].to("cpu"), 
            #                            outputs["panoptic_seg"][1])
            vis_im = Visualizer(im[..., ::-1].copy(),
                                metadata=metadata,
                                scale=1
                                )
            vis_im_gt = Visualizer(im[..., ::-1].copy(),
                                metadata=metadata,
                                scale=1
                                )
            vis_im = vis_im.draw_sem_seg(outputs["sem_seg"], alpha=0.4)
            vis_im_gt = vis_im_gt.draw_sem_seg(sem_seg_gt, alpha=0.4)
            # vis_im = vis_im.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
            # vis_im_gt = vis_im_gt.draw_panoptic_seg(pano_gt, [item | {"isthing": True} for item in inputs["segments_info"]])
            
            f = plt.get_current_fig_manager()
            f.window.showMaximized()
            f.window.setWindowTitle(inputs["file_name"])
            
            f.canvas.mpl_connect('key_press_event', keypress)
            
            ax[0].imshow(vis_im.get_image())
            ax[0].axis('off')
            ax[1].imshow(vis_im_gt.get_image())
            ax[1].axis('off')
            # f.title(inputs["file_name"])
            global _keypress_result
            _keypress_result = None
            f.canvas.draw()
            while _keypress_result is None:
                plt.waitforbuttonpress()
            if _keypress_result == "delete":
                print(i+1, f"{inputs['original_file_name']}: DELETE")
                delete_results.append(inputs['original_file_name'])
            elif _keypress_result == "bad":
                print(i+1, f"{inputs['original_file_name']}: BAD")
                bad_results.append(inputs['original_file_name'])
            else:
                print(i+1, f"{inputs['original_file_name']}")
                remaining_results.append(inputs['original_file_name'])
                
    if args.eval_path and (delete_results or bad_results):
        eval_path = Path(args.eval_path)
        assert eval_path.suffix == ".txt", "Saving should be done in \".txt\" file"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path_bad = eval_path.parent.joinpath("".join([eval_path.stem, "bad"] + eval_path.suffixes))
        eval_path_delete = eval_path.parent.joinpath("".join([eval_path.stem, "delete"] + eval_path.suffixes))
        
        with eval_path.open(mode="w") as f:
            f.writelines(line + '\n' for line in remaining_results)
        with eval_path_bad.open(mode="w") as f:
            f.writelines(line + '\n' for line in bad_results)
        with eval_path_delete.open(mode="w") as f:
            f.writelines(line + '\n' for line in delete_results)

if __name__ == "__main__":
    args = get_arguments()
    main(args)
