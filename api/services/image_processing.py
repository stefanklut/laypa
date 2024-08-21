# Imports

# > Standard Libraries
import logging
from typing import Any

# > External Libraries
import torch

# > Project Libraries
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())


def safe_predict(data, device, predict_gen_page_wrapper) -> Any:

    try:
        return predict_gen_page_wrapper.predictor(data, device)
    except Exception as exception:
        if isinstance(exception, torch.cuda.OutOfMemoryError) or (
            isinstance(exception, RuntimeError) and "NVML_SUCCESS == r INTERNAL ASSERT FAILED" in str(
                exception)
        ):
            logger.warning("CUDA OOM encountered, falling back to CPU.")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return predict_gen_page_wrapper.predictor(data, "cpu")
