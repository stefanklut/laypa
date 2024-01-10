import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from datasets.dataset import create_data


class TestCreateData(unittest.TestCase):
    @patch("json.load")
    @patch.object(Path, "open", new_callable=mock_open, read_data="test")
    def test_create_data(self, mock_path_open, mock_json):
        mock_json.return_value = {"annotations": "test", "segments_info": "test"}
        input_data = {
            "image_paths": Path("image.jpg"),
            "original_image_paths": Path("original_image.jpg"),
            "sem_seg_paths": Path("sem_seg.jpg"),
            "instances_paths": Path("instances.json"),
            "pano_paths": Path("pano.jpg"),
            "segments_info_paths": Path("segments_info.json"),
        }
        expected_output = {
            "file_name": "image.jpg",
            "original_file_name": "original_image.jpg",
            "image_id": "image",
            "sem_seg_file_name": "sem_seg.jpg",
            "annotations": "test",
            "pan_seg_file_name": "pano.jpg",
            "segments_info": "test",
        }
        with patch("pathlib.Path.is_file", return_value=True):
            self.assertEqual(create_data(input_data), expected_output)

    def test_create_data_missing_image(self):
        input_data = {
            "original_image_paths": Path("original_image.jpg"),
            "sem_seg_paths": Path("sem_seg.jpg"),
            "instances_paths": Path("instances.json"),
            "pano_paths": Path("pano.jpg"),
            "segments_info_paths": Path("segments_info.json"),
        }
        with self.assertRaises(ValueError):
            create_data(input_data)

    def test_create_data_missing_original_image(self):
        input_data = {
            "image_paths": Path("image.jpg"),
            "sem_seg_paths": Path("sem_seg.jpg"),
            "instances_paths": Path("instances.json"),
            "pano_paths": Path("pano.jpg"),
            "segments_info_paths": Path("segments_info.json"),
        }
        with self.assertRaises(ValueError):
            create_data(input_data)

    def test_create_data_file_not_found(self):
        input_data = {
            "image_paths": Path("image.jpg"),
            "original_image_paths": Path("original_image.jpg"),
            "sem_seg_paths": Path("sem_seg.jpg"),
            "instances_paths": Path("instances.json"),
            "pano_paths": Path("pano.jpg"),
            "segments_info_paths": Path("segments_info.json"),
        }
        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                create_data(input_data)


if __name__ == "__main__":
    unittest.main()
