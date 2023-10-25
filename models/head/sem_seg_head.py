from typing import Callable, Dict, Optional, Sequence, Union

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.meta_arch.semantic_seg import (
    SEM_SEG_HEADS_REGISTRY,
    SemSegFPNHead,
)
from torch.nn import functional as F


@SEM_SEG_HEADS_REGISTRY.register()
class WeightedSemSegFPNHead(SemSegFPNHead):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        weight: Sequence[float],
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
    ):
        super().__init__(
            input_shape,
            num_classes=num_classes,
            conv_dims=conv_dims,
            common_stride=common_stride,
            loss_weight=loss_weight,
            norm=norm,
            ignore_value=ignore_value,
        )
        if len(weight) != num_classes:
            raise ValueError("Number of specified weights must match the number of classes")
        self.weight = weight

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        config = super().from_config(cfg, input_shape)
        config.update({"weight": cfg.MODEL.SEM_SEG_HEAD.WEIGHT})

        return config

    def losses(self, predictions, targets):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions,
            scale_factor=self.common_stride,
            mode="bilinear",
            align_corners=False,
        )
        weight = torch.tensor(self.weight, dtype=torch.float, device=predictions.device)
        loss = F.cross_entropy(predictions, targets, weight=weight, reduction="mean", ignore_index=self.ignore_value)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
