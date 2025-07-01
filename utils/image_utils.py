import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

# import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]


# Taken from detectron2.data.detection_utils
def convert_PIL_to_numpy(image: Image.Image, format: str) -> np.ndarray:
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image_array = np.asarray(image)

    # handle formats not supported by PIL
    if format == "BGR":
        # flip channels if needed
        image_array = image_array[:, :, ::-1]
    elif format == "YUV-BT.601":
        image_array = image_array / 255.0
        image_array = np.dot(image_array, np.array(_M_RGB2YUV).T)

    return image_array


def image_to_array_dpi(
    image: Image.Image,
    mode: str,
    ignore_exif: bool,
    image_path: Path | str,
) -> tuple[np.ndarray, Optional[int]]:
    """
    Convert image to numpy array and get DPI

    Args:
        image (PIL.Image): The image to convert.
        mode (str): The color mode of the image. Supported values are "color" and "grayscale".
        ignore_exif (bool): Whether to ignore the EXIF data of the image.

    Raises:
        OSError: If the image is None after EXIF transpose.
        AssertionError: If the DPI is invalid or non-square.

    Returns:
        tuple[np.ndarray, Optional[int]]: The image as a numpy array and the DPI (dots per inch) of the image.
    """
    if not ignore_exif:
        transposed_image = ImageOps.exif_transpose(image)
        if transposed_image is None:
            raise OSError("Image is None after exif transpose")
        image = transposed_image
    dpi = image.info.get("dpi")
    if dpi is not None:
        assert len(dpi) == 2, f"Invalid DPI: {dpi}, for image: {image_path}"
        assert dpi[0] == dpi[1], f"Non-square DPI: {dpi}, for image: {image_path}"
        dpi = dpi[0]
    image_array = convert_PIL_to_numpy(image, "RGB" if mode == "color" else "L").copy()
    return image_array, dpi


def load_image_array_from_path(
    image_path: Path | str,
    mode: str = "color",
    ignore_exif: bool = False,
) -> Optional[dict]:
    """
    Load image from a given path, return None if loading failed due to corruption

    Args:
        image_path (Path | str): Path to an image on the current filesystem.
        mode (str, optional): Color mode, either "color" or "grayscale". Defaults to "color".
        ignore_exif (bool, optional): Ignore exif orientation. Defaults to False.

    Returns:
        Optional[dict]: A dictionary containing the loaded image and dpi, or None if loading failed.

    Raises:
        AssertionError: If the mode is not supported.
        AssertionError: If the DPI is invalid or non-square.

    Notes:
        - Supported image file formats: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
        - The loaded image is converted to a numpy array.

    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        image = Image.open(image_path)
        image, dpi = image_to_array_dpi(image, mode, ignore_exif, image_path)
        return {"image": image, "dpi": dpi}
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f'Cannot load image: "{image_path}" skipping for now')
        return None
    except Image.DecompressionBombError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(
            f'Trying to load image: "{image_path}" has failed due to decompression bomb error in PIL (see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open for more info). skipping for now'
        )
        return None


def load_image_array_from_bytes(
    image_bytes: bytes,
    image_path: Path,
    mode: str = "color",
    ignore_exif: bool = False,
) -> Optional[dict]:
    """
    Load an image from bytes and convert it to a numpy array.

    Args:
        image_bytes (bytes): The image bytes to load.
        image_path (Path): The path to the image file.
        mode (str, optional): The color mode of the image. Supported values are "color" and "grayscale". Defaults to "color".
        ignore_exif (bool, optional): Whether to ignore the EXIF data of the image. Defaults to False.

    Returns:
        Optional[dict]: A dictionary containing the loaded image as a numpy array and the DPI (dots per inch) of the image.
            If the image cannot be loaded, None is returned.

    Raises:
        AssertionError: If the specified mode is not supported.
        AssertionError: If the DPI is invalid or non-square.

    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        image = Image.open(BytesIO(image_bytes))
        image, dpi = image_to_array_dpi(image, mode, ignore_exif, image_path)
        return {"image": image, "dpi": dpi}
    except OSError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(f'Cannot load image: "{image_path_info}" skipping for now')
        return None
    except Image.DecompressionBombError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(
            f'Trying to load image: "{image_path_info}" has failed due to decompression bomb error in PIL (see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open for more info). skipping for now'
        )
        return None


def save_image_array_to_path(
    image_path: Path | str,
    array: np.ndarray,
    dpi: Optional[int] = None,
):
    """
    Save image to a given path, log error in case of an error

    Args:
        image_path (Path | str): The path where the image will be saved.
        array (np.ndarray): The image in array form (RGB between 0 and 255).
        dpi (Optional[int]): The DPI (dots per inch) of the saved image. Defaults to None.
    """
    try:
        # cv2.imwrite(str(image_path), array)
        image = Image.fromarray(array)
        if dpi is not None:
            image.info["dpi"] = (dpi, dpi)
        image.save(image_path)
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot save image: {image_path}, skipping for now")


if __name__ == "__main__":
    image_path = Path("./tutorial/data/inference/NL-HaNA_1.01.02_3112_0395.jpg")
    result = load_image_array_from_path(image_path, mode="color")
    if result is not None:
        image = result["image"]
    else:
        raise ValueError(f"Failed to load image from path: {image_path}")

    # image = np.zeros((100, 100)).astype(np.uint8)
    # image[25:75, 25:75] = 255
    # image_bytes = image.tobytes()

    image_bytes = image.tobytes()

    image = load_image_array_from_bytes(image_bytes, image_path, mode="color")
    if image is not None:
        print(image["image"].shape)
    else:
        raise ValueError("Failed to load image from bytes")
