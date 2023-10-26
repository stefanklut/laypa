import detectron2.data.transforms as T
import numpy as np
import torchvision.transforms.functional as F


# TODO Add all
class TorchTransform(T.Transform):
    # TODO port torch data augmentation directly into this format using functional
    def __init__(self) -> None:
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        pass
        return img

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return super().apply_segmentation(segmentation)

    def inverse(self) -> T.Transform:
        raise NotImplementedError
