import argparse
from datasets.augmentations import ResizeShortestEdge
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
import matplotlib.pyplot as plt
import cv2
from main import preprocess_datasets, setup_cfg
import torch
from natsort import os_sorted
from utils.tempdir import OptionalTemporaryDirectory


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

    args = parser.parse_args()

    return args


class Predictor(DefaultPredictor):
    """
    Predictor runs the model specified in the config, on call the image is processed and the results dict is output
    """
    def __init__(self, cfg):
        """
        Predictor runs the model specified in the config, on call the image is processed and the results dict is output

        Args:
            cfg (CfgNode): config
        """
        super().__init__(cfg)

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.TEST.WEIGHTS)

        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST,
        )

    def __call__(self, original_image):
        """
        Not really useful, but shows what call needs to be made
        """
        return super().__call__(original_image)


def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    # Setup config
    cfg = setup_cfg(args, save_config=False)

    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not(args.keep_tmp_dir)) as tmp_dir:
        
        preprocess_datasets(cfg, args.train, args.val, tmp_dir, save_image_locations=False)
        predictor = Predictor(cfg=cfg)

        # train_loader = DatasetCatalog.get("train")
        val_loader = DatasetCatalog.get("val")
        metadata = MetadataCatalog.get("val")
        
        # for inputs in np.random.choice(val_loader, 3):
        for inputs in os_sorted(val_loader, key=lambda x: x["file_name"]):
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
            ax[0].imshow(vis_im.get_image())
            ax[0].axis('off')
            ax[1].imshow(vis_im_gt.get_image())
            ax[1].axis('off')
            # f.title(inputs["file_name"])
            plt.show()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
