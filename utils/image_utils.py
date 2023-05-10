from io import BytesIO
import logging
from typing import Optional
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import sys
import torchvision

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger_name

def load_image_from_path(image_path: Path | str, mode="color") -> Optional[np.ndarray]:
    """
    Load image from a given path, return None if loading failed due to corruption

    Args:
        image_path (Path | str): path to an image on current filesystem

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    #Supported: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    
    if mode == "color":
        conversion = cv2.COLOR_RGB2BGR
    elif mode == "grayscale":
        conversion = cv2.COLOR_RGB2GRAY
    else:
        raise NotImplementedError
    
    try:
        # image = Image.open(image_path)
        # image = cv2.cvtColor(np.asarray(image), conversion)
        # image = torchvision.io.read_image(str(image_path)).permute(1,2,0).numpy()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE)
        return image
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path} skipping for now")
        return None
    

def load_image_from_bytes(img_bytes: bytes, image_path: Optional[Path]=None) -> Optional[np.ndarray]:
    """
    Load image based on given bytes, return None if loading failed due to corruption

    Args:
        img_bytes (bytes): transfer bytes of data that represent an image
        image_path (Optional[Path], optional): image_path for logging. Defaults to None.

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    try:
        # bytes_array = np.frombuffer(img_bytes, np.uint8)
        # image = cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)
        image = Image.open(BytesIO(img_bytes))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
    except OSError:
        image_path_info = image_path if image_path is not None else "Filename not given"
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot load image: {image_path_info} skipping for now")
        return None
    
def save_image_to_path(image_path: Path | str, array: np.ndarray):
    """
    Save image to a given path, log error in case of an error

    Args:
        image_path (Path | str): save path location
        array (np.ndarray): image in array form (BGR between 0 and 255)
    """
    try:
        # cv2.imwrite(str(image_path), array)
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(array)
        image.save(image_path)
    except OSError:
        logger = logging.getLogger(get_logger_name())
        logger.warning(f"Cannot save image: {image_path}, skipping for now")
    
    
if __name__ == "__main__":
    # image_path = Path("/home/stefan/Downloads/corrupt.png")
    # output = load_image_from_path("test.png", mode="grayscale")
    # print(output.shape)
    image = np.zeros((100,100))
    image[25:75, 25:75] = 255
    save_image_to_path("test.png", image)   
