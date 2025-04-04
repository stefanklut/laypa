import copy
import logging
from typing import Any, Optional, override

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.config import CfgNode, configurable
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import (
    SizeMismatchError,
    check_image_size,
    create_keypoint_hflip_indices,
    transform_proposals,
)
from detectron2.data.transforms.augmentation import _check_img_dtype

from data.augmentations import build_augmentation
from utils.array_utils import load_array
from utils.logging_utils import get_logger_name


class _TransformToAug(T.Augmentation):
    def __init__(self, tfm: T.Transform):
        self.tfm = tfm

    def get_transform(self, *args):
        return self.tfm

    def __repr__(self):
        return repr(self.tfm)

    __str__ = __repr__


def _transform_to_aug(tfm_or_aug):
    """
    Wrap Transform into Augmentation.
    Private, used internally to implement augmentations.
    """
    assert isinstance(tfm_or_aug, (T.Transform, T.Augmentation)), tfm_or_aug
    if isinstance(tfm_or_aug, T.Augmentation):
        return tfm_or_aug
    else:
        return _TransformToAug(tfm_or_aug)


def _check_img_dtype(img: np.ndarray | torch.Tensor) -> None:
    """
    Check if the image is a numpy array or torch tensor.

    Args:
        img: The image to check.

    Raises:
        ValueError: If the image is not a numpy array or torch tensor.
    """
    if isinstance(img, torch.Tensor):
        assert img.dtype == torch.uint8 or img.dtype == torch.float32, f"[Augmentation] Got image of type {img.dtype}!"
        assert img.dim() == 3, img.dim()
    elif isinstance(img, np.ndarray):
        assert img.dtype == np.uint8 or img.dtype == np.float32, f"[Augmentation] Got image of type {img.dtype}!"
        assert img.ndim in [2, 3], img.ndim
    else:
        raise ValueError("[Augmentation] Needs an numpy array or torch tensor, but got a {}!".format(type(img)))


class AugInput(T.AugInput):
    """
    A class that holds input data for augmentation.
    """

    def __init__(
        self,
        image: np.ndarray | torch.Tensor,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray | torch.Tensor] = None,
        dpi: Optional[int] = None,
        auto_dpi: Optional[bool] = False,
        default_dpi: Optional[int] = None,
        manual_dpi: Optional[int] = None,
    ):
        """
        Initialize the AugInput class.

        Args:
            image (np.ndarray | torch.Tensor): The image to augment.
            boxes (Optional[np.ndarray], optional): The boxes to augment. Defaults to None.
            sem_seg (Optional[np.ndarray  |  torch.Tensor], optional): The semantic segmentation to augment. Defaults to None.
            dpi (Optional[int], optional): The DPI of the image. Defaults to None.
            auto_dpi (Optional[bool], optional): Whether to automatically detect the DPI. Defaults to False.
            default_dpi (Optional[int], optional): The default DPI to use. Defaults to None.
            manual_dpi (Optional[int], optional): The manual DPI to use. Defaults to None.
        """

        _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg
        if auto_dpi:
            if dpi is None:
                self.dpi = default_dpi
            else:
                self.dpi = dpi
        else:
            self.dpi = manual_dpi

    # def transform(self, tfm: T.Transform) -> None:
    #     """
    #     In-place transform all attributes of this class.

    #     By "in-place", it means after calling this method, accessing an attribute such
    #     as ``self.image`` will return transformed data.
    #     """
    #     self.image = tfm.apply_image(self.image)
    #     if self.boxes is not None:
    #         self.boxes = tfm.apply_box(self.boxes)
    #     if self.sem_seg is not None:
    #         self.sem_seg = tfm.apply_segmentation(self.sem_seg)

    def apply_augmentations(self, augmentations: list[T.Augmentation | T.Transform]) -> T.TransformList:
        """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
        return T.AugmentationList(augmentations)(self)


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.

    If height or width is not specified in the dict, it will be filled with the image's shape.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        if isinstance(image, torch.Tensor):
            image_wh = (image.shape[-1], image.shape[-2])
        else:
            image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    (" for image " + dataset_dict["file_name"] if "file_name" in dataset_dict else ""),
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        if isinstance(image, torch.Tensor):
            dataset_dict["width"] = image.shape[-1]
        else:
            dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        if isinstance(image, torch.Tensor):
            dataset_dict["height"] = image.shape[-2]
        else:
            dataset_dict["height"] = image.shape[0]


class Mapper(DatasetMapper):
    """
    Base class for mappers that handle data augmentation.
    """

    @configurable
    def __init__(
        self,
        mode: str = "train",
        *,
        augmentations: list[T.Augmentation | T.Transform],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        auto_dpi: Optional[bool] = True,
        default_dpi: Optional[int] = None,
        manual_dpi: Optional[int] = None,
        on_gpu: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        assert mode in ["train", "val", "test"], f"Unknown mode: {mode}"
        is_train = True if mode == "train" else False

        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"

        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk = precomputed_proposal_topk
        self.recompute_boxes = recompute_boxes
        self.auto_dpi = auto_dpi
        self.default_dpi = default_dpi
        self.manual_dpi = manual_dpi
        self.on_gpu = on_gpu
        self.device = device

        logger = logging.getLogger(get_logger_name())
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg: CfgNode, mode: str = "train", device=torch.device("cpu")) -> dict[str, Any]:
        """
        Converts a configuration object to a dictionary to be used as keyword arguments.

        Args:
            cfg (CfgNode): The configuration object.

        Returns:
            dict[str, Any]: A dictionary containing the converted configuration values.
        """
        augs = build_augmentation(cfg, mode)

        if cfg.INPUT.CROP.ENABLED and mode == "train":
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "mode": mode,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "auto_dpi": cfg.INPUT.DPI.AUTO_DETECT_TRAIN,
            "default_dpi": cfg.INPUT.DPI.DEFAULT_DPI_TRAIN,
            "manual_dpi": cfg.INPUT.DPI.MANUAL_DPI_TRAIN,
            "on_gpu": cfg.INPUT.ON_GPU,
            "device": device,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN if mode == "train" else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def __call__(self, dataset_dict):
        raise NotImplementedError("Mapper must be subclassed")


class SemSegMapper(Mapper):
    """
    Specialized mapper for semantic segmentation tasks.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load image.
        image = load_array(dataset_dict["image_file_name"], mode="color", on_gpu=self.on_gpu, device=self.device)

        check_image_size(dataset_dict, image["image"])

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = load_array(dataset_dict["sem_seg_file_name"], mode="grayscale", on_gpu=self.on_gpu, device=self.device)
        else:
            sem_seg_gt = {"image": None, "dpi": None}

        assert type(image["image"]) == type(
            sem_seg_gt["image"]
        ), f"Image and sem_seg_gt have different types: {type(image['image'])} and {type(sem_seg_gt['image'])}"

        aug_input = AugInput(
            image["image"],
            sem_seg=sem_seg_gt["image"],
            dpi=image["dpi"],
            auto_dpi=self.auto_dpi,
            default_dpi=self.default_dpi,
            manual_dpi=self.manual_dpi,
        )
        with torch.no_grad():
            transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        if image is None:
            raise ValueError(f"Image {dataset_dict['image_file_name']} has become None after augmentation")

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if isinstance(image, torch.Tensor):
            image_shape = image.shape[-2:]  # h, w
            dataset_dict["image"] = image.clone()
        elif isinstance(image, np.ndarray):
            image_shape = image.shape[:2]
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            raise ValueError(f"image is not a numpy array or torch tensor: {type(image)}")

        if sem_seg_gt is not None:
            if isinstance(sem_seg_gt, torch.Tensor):
                dataset_dict["sem_seg"] = sem_seg_gt.to(dtype=torch.long).squeeze(0).clone()
            elif isinstance(sem_seg_gt, np.ndarray):
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
            else:
                raise ValueError(f"sem_seg_gt is not a numpy array or torch tensor: {type(sem_seg_gt)}")

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        return dataset_dict


class SemSegInstancesMapper(Mapper):
    """
    Specialized mapper for semantic segmentation tasks with instance annotations.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load image.
        image = load_array(dataset_dict["image_file_name"], mode="color", on_gpu=self.on_gpu, device=self.device)

        check_image_size(dataset_dict, image["image"])

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = load_array(dataset_dict["sem_seg_file_name"], mode="grayscale", on_gpu=self.on_gpu, device=self.device)
        else:
            sem_seg_gt = {"image": None, "dpi": None}

        assert type(image["image"]) == type(
            sem_seg_gt["image"]
        ), f"Image and sem_seg_gt have different types: {type(image['image'])} and {type(sem_seg_gt['image'])}"

        aug_input = AugInput(
            image["image"],
            sem_seg=sem_seg_gt["image"],
            dpi=image["dpi"],
            auto_dpi=self.auto_dpi,
            default_dpi=self.default_dpi,
            manual_dpi=self.manual_dpi,
        )
        with torch.no_grad():
            transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        if image is None:
            raise ValueError(f"Image {dataset_dict['image_file_name']} has become None after augmentation")

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if isinstance(image, torch.Tensor):
            image_shape = image.shape[-2:]  # h, w
            dataset_dict["image"] = image.clone()
        elif isinstance(image, np.ndarray):
            image_shape = image.shape[:2]
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            raise ValueError(f"image is not a numpy array or torch tensor: {type(image)}")

        if sem_seg_gt is not None:
            if isinstance(sem_seg_gt, torch.Tensor):
                dataset_dict["sem_seg"] = sem_seg_gt.to(dtype=torch.long).squeeze(0).clone()
            elif isinstance(sem_seg_gt, np.ndarray):
                dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
            else:
                raise ValueError(f"sem_seg_gt is not a numpy array or torch tensor: {type(sem_seg_gt)}")

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class BinarySegMapper(Mapper):
    """
    Specialized mapper for binary segmentation tasks.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # Load image.
        image = load_array(dataset_dict["image_file_name"], mode="color")

        check_image_size(dataset_dict, image["image"])

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "binary_seg_file_name" in dataset_dict:
            binary_seg_gt = load_array(
                dataset_dict["binary_seg_file_name"], mode="color", on_gpu=self.on_gpu, device=self.device
            )
        else:
            binary_seg_gt = {"image": None, "dpi": None}

        assert type(image["image"]) == type(
            binary_seg_gt["image"]
        ), f"Image and sem_seg_gt have different types: {type(image['image'])} and {type(binary_seg_gt['image'])}"

        aug_input = AugInput(
            image["image"],
            sem_seg=binary_seg_gt["image"],
            dpi=image["dpi"],
            auto_dpi=self.auto_dpi,
            default_dpi=self.default_dpi,
            manual_dpi=self.manual_dpi,
        )
        with torch.no_grad():
            transforms = self.augmentations(aug_input)
        image, binary_seg_gt = aug_input.image, aug_input.sem_seg

        if image is None:
            raise ValueError(f"Image {dataset_dict['image_file_name']} has become None after augmentation")

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if isinstance(image, torch.Tensor):
            dataset_dict["image"] = image.clone()
        elif isinstance(image, np.ndarray):
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            raise ValueError(f"image is not a numpy array or torch tensor: {type(image)}")

        if binary_seg_gt is not None:
            if isinstance(binary_seg_gt, torch.Tensor):
                dataset_dict["binary_seg"] = binary_seg_gt.to(dtype=torch.long).clone()
            elif isinstance(binary_seg_gt, np.ndarray):
                dataset_dict["binary_seg"] = torch.as_tensor(binary_seg_gt.astype("long").transpose(2, 0, 1))
            else:
                raise ValueError(f"sem_seg_gt is not a numpy array or torch tensor: {type(binary_seg_gt)}")

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("binary_seg_file_name", None)
            return dataset_dict

        return dataset_dict
