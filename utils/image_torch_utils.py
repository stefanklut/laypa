import logging
import sys
from pathlib import Path
from typing import Optional

import imagesize
import torch
import torchvision

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name


def load_image_tensor_from_path(
    path: Path,
    mode: str = "color",
    device: torch.device = torch.device("cpu"),
    ignore_exif: bool = False,
) -> Optional[dict]:
    """
    Load an image from a given path and convert it to a torch tensor.

    Args:
        path (Path): The path to the image file.
        mode (str, optional): The color mode of the image. Supported values are "color" and "grayscale". Defaults to "color".
        device (torch.device, optional): The device to load the image tensor to. Defaults to torch.device("cpu").
        ignore_exif (bool, optional): Whether to ignore the EXIF data of the image. Defaults to False.

    Returns:
        Optional[dict]: _description_
    """
    assert mode in ["color", "grayscale"], f'Mode "{mode}" not supported'

    try:
        torchvision_mode = torchvision.io.ImageReadMode.RGB if mode == "color" else torchvision.io.ImageReadMode.GRAY

        if path.suffix in [".JPG", ".JPEG", ".jpeg", ".jpg"]:
            image_bytes = torchvision.io.read_file(str(path))
            image = torchvision.io.decode_jpeg(
                image_bytes,
                mode=torchvision_mode,
                device=device,  # type: ignore
                apply_exif_orientation=not ignore_exif,
            )
        else:
            image = torchvision.io.read_image(
                str(path),
                torchvision_mode,
                apply_exif_orientation=not ignore_exif,
            )
            image = image.to(device)
        dpi = imagesize.getDPI(path)
        assert len(dpi) == 2, f"Invalid DPI: {dpi}"
        assert dpi[0] == dpi[1], f"Non-square DPI: {dpi}"
        if dpi == (-1, -1):
            dpi = None
        else:
            dpi = dpi[0]

        return {"image": image, "dpi": dpi}
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {path} skipping for now")
        return None


if __name__ == "__main__":
    image = load_image_tensor_from_path(
        Path("~/Documents/datasets/mini-republic/train/NL-HaNA_1.01.02_62_0109.jpg").expanduser(),
        mode="color",
        device=torch.device("cuda"),
    )

    print(image["image"].shape)
    print(image["dpi"])
    print(image["image"].dtype)
    print(image["image"].device)
