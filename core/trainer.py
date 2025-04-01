import copy
import itertools
import logging
import os
import weakref
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Set

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    TrainerBase,
    create_ddp_model,
    hooks,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping, reduce_param_groups
from detectron2.utils import comm

from data.mapper import BinarySegMapper, SemSegInstancesMapper, SemSegMapper
from evaluation.binary_seg_evaluation import BinarySegEvaluator
from evaluation.sem_seg_evaluation import SemSegEvaluator
from utils.logging_utils import get_logger_name


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr: Optional[float] = None,
    backbone_multiplier: Optional[float] = 1.0,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    weight_decay_embed: Optional[float] = None,
    bias_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = None,
    lr_factor_func: Optional[Callable] = None,
    overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters.
        lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
            corresponding lr decay rate. Note that setting this option requires
            also setting ``base_lr``.
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides
    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)

            if "backbone" in module_name:
                hyperparams["lr"] = hyperparams["lr"] * backbone_multiplier

            if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                hyperparams["weight_decay"] = 0.0

            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = weight_decay_embed

            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm

            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        backbone_multiplier=cfg.SOLVER.BACKBONE_MULTIPLIER,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        weight_decay_embed=cfg.SOLVER.WEIGHT_DECAY_EMBED,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )

    def maybe_add_full_model_gradient_clipping(optimizer):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optimizer):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optimizer

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif optimizer_type == "ADAM":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.AMSGRAD,
        )
    elif optimizer_type == "ADAMW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            amsgrad=cfg.SOLVER.AMSGRAD,
        )
    elif optimizer_type == "ADAGRAD":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adagrad)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)  # type: ignore
    return optimizer  # type: ignore


MetaArchitechture_converter: dict[str, dict[str, Any]] = {
    "SemanticSegmentor": {
        "mapper": SemSegMapper,
        "evaluator": SemSegEvaluator,
        "output": "sem_seg",
    },
    "MaskFormer": {
        "mapper": SemSegInstancesMapper,
        "evaluator": SemSegEvaluator,
        "output": "sem_seg",
    },
    "BinarySegmentor": {
        "mapper": BinarySegMapper,
        "evaluator": BinarySegEvaluator,
        "output": "binary_seg",
    },
}


class Trainer(DefaultTrainer):
    """
    Trainer class
    """

    def __init__(self, cfg: CfgNode, validation: bool = False):
        TrainerBase.__init__(self)

        # logger = logging.getLogger("detectron2")
        # if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
        #     setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        model = create_ddp_model(model, broadcast_buffers=False)

        data_loader = self.build_train_loader(cfg, device=model.device) if not validation else None

        self._trainer = (AMPTrainer if cfg.MODEL.AMP_TRAIN.ENABLED else SimpleTrainer)(model, data_loader, optimizer)
        if isinstance(self._trainer, AMPTrainer):
            precision_converter = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            precision = precision_converter.get(cfg.MODEL.AMP_TRAIN.PRECISION, None)
            if precision is None:
                raise ValueError(f"Unrecognized precision: {cfg.MODEL.AMP_TRAIN.PRECISION}")
            self._trainer.precision = precision

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        checkpoint_save_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        self.checkpointer = DetectionCheckpointer(
            model,
            checkpoint_save_dir,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        output = MetaArchitechture_converter[cfg.MODEL.META_ARCHITECTURE]["output"]

        miou_checkpointer = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=f"{output}/mIoU",
            mode="max",
            file_prefix="model_best_mIoU",
        )

        fwiou_checkpointer = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=f"{output}/fwIoU",
            mode="max",
            file_prefix="model_best_fwIoU",
        )

        macc_checkpointer = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=f"{output}/mACC",
            mode="max",
            file_prefix="model_best_mACC",
        )

        pacc_checkpointer = hooks.BestCheckpointer(
            eval_period=cfg.TEST.EVAL_PERIOD,
            checkpointer=self.checkpointer,
            val_metric=f"{output}/pACC",
            mode="max",
            file_prefix="model_best_pACC",
        )

        self.register_hooks(self.build_hooks() + [miou_checkpointer, fwiou_checkpointer, macc_checkpointer, pacc_checkpointer])

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator = MetaArchitechture_converter[cfg.MODEL.META_ARCHITECTURE]["evaluator"](
            dataset_name,
            distributed=True,
        )

        return evaluator

    @classmethod
    def get_mapper(cls, cfg, device=torch.device("cpu"), mode="train"):
        mapper = MetaArchitechture_converter[cfg.MODEL.META_ARCHITECTURE]["mapper"](
            cfg,
            mode=mode,
            on_gpu=cfg.INPUT.ON_GPU,
            device=device,
        )

        return mapper

    @classmethod
    def build_train_loader(cls, cfg, device=torch.device("cpu")):
        mapper = cls.get_mapper(cfg, device=device, mode="train")

        return build_detection_train_loader(cfg=cfg, mapper=mapper, pin_memory=cfg.DATALOADER.PIN_MEMORY)  # type: ignore

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, device=torch.device("cpu")):
        mapper = cls.get_mapper(cfg, device=device, mode="val")

        return build_detection_test_loader(cfg=cfg, mapper=mapper, dataset_name=dataset_name)  # type: ignore

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(get_logger_name())
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(len(cfg.DATASETS.TEST), len(evaluators))

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, model.device)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(results_i, dict), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def validate(self):
        results = self.test(self.cfg, self.model)  # type: ignore
