from io import BytesIO
from typing import Optional
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

def load_image_from_path(image_path: Path | str) -> Optional[np.ndarray]:
    """
    Load image from a given path, return None if loading failed due to corruption

    Args:
        image_path (Path | str): path to an image on current filesystem

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    #Supported: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    try:
        image = Image.open(image_path)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
    except OSError:
        return None
    

def load_image_from_bytes(img_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load image based on given bytes, return None if loading failed due to corruption

    Args:
        img_bytes (bytes): transfer bytes of data that represent an image

    Returns:
        Optional[np.ndarray]: the loaded image or None
    """
    try:
        image = Image.open(BytesIO(img_bytes))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
    except OSError:
        return None
    
    
if __name__ == "__main__":
    image_path = Path("/home/stefan/Downloads/corrupt.png")
    load_image_from_path(image_path)