import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision
from detectron2.data.detection_utils import convert_PIL_to_numpy
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name


def load_image_array_from_path(
    image_path: Path | str,
    mode: str = "color",
    ignore_exif: bool = False,
) -> Optional[np.ndarray]:
    """
    Load image from a given path, return None if loading failed due to corruption

    Args:
        image_path (Path | str): path to an image on current filesystem
        mode (str): color mode, either "color" or "grayscale"

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    # Supported: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        # image = cv2.imread(str(image_path), cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if mode == "color" else image
        image = Image.open(image_path)
        if not ignore_exif:
            image = ImageOps.exif_transpose(image)
        image = convert_PIL_to_numpy(image, "RGB" if mode == "color" else "L").copy()
        if mode == "grayscale":
            image = image.squeeze(axis=2)
        return image
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path} skipping for now")
        return None


def load_image_tensor_from_path(
    image_path: Path | str,
    mode: str = "color",
) -> Optional[torch.Tensor]:
    """
    Load image from a given path, return None if loading failed due to corruption

    Args:
        image_path (Path | str): path to an image on current filesystem
        mode (str): color mode, either "color" or "grayscale"

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        image = torchvision.io.read_image(
            str(image_path), torchvision.io.ImageReadMode.RGB if mode == "color" else torchvision.io.ImageReadMode.GRAY
        )
        return image
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path} skipping for now")
        return None


def load_image_array_from_bytes(
    image_bytes: bytes,
    image_path: Optional[Path] = None,
    mode: str = "color",
    ignore_exif: bool = False,
) -> Optional[np.ndarray]:
    """
    Load image based on given bytes, return None if loading failed due to corruption

    Args:
        image_bytes (bytes): transfer bytes of data that represent an image
        image_path (Optional[Path], optional): image_path for logging. Defaults to None.
        mode (str, optional): color mode, either "color" or "grayscale". Defaults to "color"

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """

    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        # bytes_array = np.frombuffer(image_bytes, np.uint8)
        # image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if mode == "color" else image
        image = Image.open(BytesIO(image_bytes))
        if not ignore_exif:
            image = ImageOps.exif_transpose(image)
        image = convert_PIL_to_numpy(image, "RGB" if mode == "color" else "L").copy()
        if mode == "grayscale":
            image = image.squeeze(axis=2)
        return image
    except OSError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path_info}. skipping for now")
        return None


def load_image_tensor_from_bytes(
    image_bytes: bytes,
    image_path: Optional[Path] = None,
    mode: str = "color",
) -> Optional[torch.Tensor]:
    """
    Load image based on given bytes, return None if loading failed due to corruption

    Args:
        image_bytes (bytes): transfer bytes of data that represent an image
        image_path (Optional[Path], optional): image_path for logging. Defaults to None.
        mode (str, optional): color mode, either "color" or "grayscale". Defaults to "color"

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        tensor = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        image = torchvision.io.decode_image(
            tensor, torchvision.io.ImageReadMode.RGB if mode == "color" else torchvision.io.ImageReadMode.GRAY
        )
        return image
    except OSError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path_info}. skipping for now")
        return None


def save_image_array_to_path(
    image_path: Path | str,
    array: np.ndarray,
):
    """
    Save image to a given path, log error in case of an error

    Args:
        image_path (Path | str): save path location
        array (np.ndarray): image in array form (RGB between 0 and 255)
    """
    try:
        # cv2.imwrite(str(image_path), array)
        image = Image.fromarray(array)
        image.save(image_path)
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot save image: {image_path}, skipping for now")


if __name__ == "__main__":
    image_path = Path("./tutorial/data/inference/NL-HaNA_1.01.02_3112_0395.jpg")
    image = load_image_array_from_path(image_path, mode="color")

    # image = np.zeros((100, 100)).astype(np.uint8)
    # image[25:75, 25:75] = 255
    # image_bytes = image.tobytes()

    image_bytes = image.tobytes()

    image = load_image_array_from_bytes(image_bytes)
    print(image.shape)
