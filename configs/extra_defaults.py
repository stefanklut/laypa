from detectron2.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.RESUME = False

_C.TRAIN = CN()
_C.TRAIN.WEIGHTS = ""

_C.TEST = CN()
_C.TEST.WEIGHTS = ""


