import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import imagesize
import torch
import torchvision
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name


def load_image_tensor_from_path_gpu_decode(
    image_path: str | Path,
    mode: str = "color",
    device: torch.device = torch.device("cpu"),
    ignore_exif: bool = False,
) -> Optional[dict]:
    """
    Load an image from a given path and convert it to a torch tensor.

    Args:
        image_path (Path): The path to the image file.
        mode (str, optional): The color mode of the image. Supported values are "color" and "grayscale". Defaults to "color".
        device (torch.device, optional): The device to load the image tensor to. Defaults to torch.device("cpu").
        ignore_exif (bool, optional): Whether to ignore the EXIF data of the image. Defaults to False.

    Returns:
        Optional[dict]: _description_
    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'
    image_path = Path(image_path)

    try:
        torchvision_mode = torchvision.io.ImageReadMode.RGB if mode == "color" else torchvision.io.ImageReadMode.GRAY
        if image_path.suffix in [".JPG", ".JPEG", ".jpeg", ".jpg"]:
            image_bytes = torchvision.io.read_file(str(image_path))
            image = torchvision.io.decode_jpeg(
                image_bytes,
                mode=torchvision_mode,
                device=device,  # type: ignore
                apply_exif_orientation=not ignore_exif,
            )
        else:
            image = torchvision.io.read_image(
                str(image_path),
                torchvision_mode,
                apply_exif_orientation=not ignore_exif,
            )
            image = image.to(device)
        dpi = imagesize.getDPI(image_path)
        assert len(dpi) == 2, f"Invalid DPI: {dpi}, for image: {image_path}"
        assert dpi[0] == dpi[1], f"Non-square DPI: {dpi}, for image: {image_path}"
        if dpi == (-1, -1):
            dpi = None
        else:
            dpi = dpi[0]

        return {"image": image, "dpi": dpi}
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path} skipping for now")
        return None


def load_image_tensor_from_path(
    image_path: Path | str,
    mode: str = "color",
) -> Optional[dict]:
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

        dpi = imagesize.getDPI(image_path)
        assert len(dpi) == 2, f"Invalid DPI: {dpi}, for image: {image_path}"
        assert dpi[0] == dpi[1], f"Non-square DPI: {dpi}, for image: {image_path}"
        if dpi == (-1, -1):
            dpi = None
        else:
            dpi = dpi[0]

        return {"image": image, "dpi": dpi}
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path} skipping for now")
        return None


def load_image_tensor_from_bytes(
    image_bytes: bytes,
    image_path: Path,
    mode: str = "color",
) -> Optional[dict]:
    """
    Load image based on given bytes, return None if loading failed due to corruption

    Args:
        image_bytes (bytes): transfer bytes of data that represent an image
        image_path (Path): image_path for logging.
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
        image_dpi = Image.open(BytesIO(image_bytes))
        dpi = image_dpi.info.get("dpi")
        if dpi is not None:
            assert len(dpi) == 2, f"Invalid DPI: {dpi}, for image: {image_path}"
            assert dpi[0] == dpi[1], f"Non-square DPI: {dpi}, for image: {image_path}"
            dpi = dpi[0]
        return {"image": image, "dpi": dpi}
    except OSError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path_info}. skipping for now")
        return None


if __name__ == "__main__":
    image = load_image_tensor_from_path_gpu_decode(
        Path("~/Documents/datasets/mini-republic/train/NL-HaNA_1.01.02_62_0109.jpg").expanduser(),
        mode="color",
        device=torch.device("cuda"),
    )
    if image is not None:
        print(image["image"].shape)
        print(image["dpi"])
        print(image["image"].dtype)
        print(image["image"].device)
    else:
        raise ValueError("Failed to load image from path")
