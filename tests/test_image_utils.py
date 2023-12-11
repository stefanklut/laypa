import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.image_utils import (
    load_image_array_from_bytes,
    load_image_array_from_path,
    save_image_array_to_path,
)


class TestImageUtils(unittest.TestCase):
    def setUp(self):
        # Tempdir for working directory of tests
        self._tmp_dir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent, prefix=".tmp", suffix=self.__class__.__name__
        )
        self.tmp_dir = Path(self._tmp_dir.name)
        self.data_dir = Path(__file__).parents[1].joinpath("tutorial", "data")
        assert self.data_dir.is_dir(), f"Missing tutorial data {self.data_dir}"

    def test_load_color_from_path(self):
        image_path = self.data_dir.joinpath("inference", "NL-HaNA_1.01.02_3112_0395.jpg")
        image = load_image_array_from_path(image_path, mode="color")
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[-1], 3)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)
        self.assertIsInstance(image[0, 0, 0], np.uint8)

    def test_load_grayscale_from_path(self):
        image_path = self.data_dir.joinpath("inference", "NL-HaNA_1.01.02_3112_0395.jpg")
        image = load_image_array_from_path(image_path, mode="grayscale")
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 2)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)
        self.assertIsInstance(image[0, 0], np.uint8)

    def test_load_color_from_bytes(self):
        image_path = self.data_dir.joinpath("inference", "NL-HaNA_1.01.02_3112_0395.jpg")
        with image_path.open(mode="rb") as f:
            image_bytes = f.read()
        image = load_image_array_from_bytes(image_bytes, image_path, mode="color")
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[-1], 3)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)
        self.assertIsInstance(image[0, 0, 0], np.uint8)

    def test_load_grayscale_from_bytes(self):
        image_path = self.data_dir.joinpath("inference", "NL-HaNA_1.01.02_3112_0395.jpg")
        with image_path.open(mode="rb") as f:
            image_bytes = f.read()
        image = load_image_array_from_bytes(image_bytes, image_path, mode="grayscale")
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.ndim, 2)
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 255)
        self.assertIsInstance(image[0, 0], np.uint8)

    def test_save_color_jpg(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.jpg")
        save_image_array_to_path(image_path, image)
        self.assertTrue(image_path.is_file())

    def test_save_grayscale_jpg(self):
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.jpg")
        save_image_array_to_path(image_path, image)
        self.assertTrue(image_path.is_file())

    def test_save_load_color_jpg(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.jpg")
        save_image_array_to_path(image_path, image)
        image2 = load_image_array_from_path(image_path, mode="color")
        self.assertIsNotNone(image2)
        self.assertTrue(np.array_equal(image, image2, equal_nan=True))

    def test_save_load_grayscale_jpg(self):
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.jpg")
        save_image_array_to_path(image_path, image)
        image2 = load_image_array_from_path(image_path, mode="grayscale")
        self.assertIsNotNone(image2)
        self.assertTrue(np.array_equal(image, image2, equal_nan=True))

    def test_save_color_png(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.png")
        save_image_array_to_path(image_path, image)
        self.assertTrue(image_path.is_file())

    def test_save_grayscale_png(self):
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.png")
        save_image_array_to_path(image_path, image)
        self.assertTrue(image_path.is_file())

    def test_save_load_color_png(self):
        image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.png")
        save_image_array_to_path(image_path, image)
        image2 = load_image_array_from_path(image_path, mode="color")
        self.assertIsNotNone(image2)
        self.assertTrue(np.array_equal(image, image2, equal_nan=True))

    def test_save_load_grayscale_png(self):
        image = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.png")
        save_image_array_to_path(image_path, image)
        image2 = load_image_array_from_path(image_path, mode="grayscale")
        self.assertIsNotNone(image2)
        self.assertTrue(np.array_equal(image, image2, equal_nan=True))

    def test_corrupt_image(self):
        image_path = self.tmp_dir.joinpath(f"{uuid.uuid4()}.jpg")
        image = load_image_array_from_path(image_path)
        self.assertIsNone(image)


if __name__ == "__main__":
    unittest.main()
