from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.image_torch_utils import load_image_tensor_from_path_gpu_decode
from utils.image_utils import load_image_array_from_path


def load_array(
    path: Path | str, mode: str = "color", on_gpu: bool = False, device: torch.device = torch.device("cpu")
) -> dict[str, Any]:
    """
    Load an array or image from a file path.

    Args:
        path (str): The path to the array or image file.
        mode (str): The mode to use when loading the image.

    Returns:
        dict: The loaded image and its DPI.
    """
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path)
        if array is None:
            raise ValueError(f"Array {path} cannot be loaded")
        assert array.ndim == 3 or array.ndim == 2, f"Invalid array shape: {array.shape}"
        if array.ndim == 2:
            array = array[:, :, None]
        return {"image": array, "dpi": None}
    elif path.suffix == ".pt":
        tensor = torch.load(path, weights_only=True)
        if tensor is None:
            raise ValueError(f"Tensor {path} cannot be loaded")

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 3:
            pass
        else:
            raise ValueError(f"Invalid tensor shape: {tensor.shape}")

        return {"image": tensor, "dpi": None}
    else:
        if on_gpu:
            data = load_image_tensor_from_path_gpu_decode(path, mode=mode, device=device)
            if data is None:
                raise ValueError(f"Image {path} cannot be loaded")
            return data
        else:
            data = load_image_array_from_path(path, mode=mode)
            if data is None:
                raise ValueError(f"Image {path} cannot be loaded")
            return data
