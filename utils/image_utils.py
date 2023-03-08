from io import BytesIO
from typing import Optional
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

def load_image_from_path(image_path: Path) -> Optional[np.ndarray]:
    #Supported: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    try:
        image = Image.open(image_path)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
    except OSError:
        return None
    

def load_image_from_bytes(img_bytes: bytes):
    try:
        image = Image.open(BytesIO(img_bytes))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return image
    except OSError:
        return None
    
    
if __name__ == "__main__":
    image_path = Path("/home/stefan/Downloads/corrupt.png")
    load_image_from_path(image_path)