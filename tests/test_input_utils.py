import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.input_utils import clean_input_paths


class TestInputUtils(unittest.TestCase):
    def test_single_string_input(self):
        input_path = "/path/to/single/file.jpg"
        expected_output = [Path("/path/to/single/file.jpg")]
        self.assertEqual(clean_input_paths(input_path), expected_output)

    def test_single_path_input(self):
        input_path = Path("/path/to/single/file.jpg")
        expected_output = [Path("/path/to/single/file.jpg")]
        self.assertEqual(clean_input_paths(input_path), expected_output)

    def test_sequence_string_input(self):
        input_paths = ["/path/to/file1.jpg", "/path/to/file2.jpg"]
        expected_output = [Path("/path/to/file1.jpg"), Path("/path/to/file2.jpg")]
        self.assertEqual(clean_input_paths(input_paths), expected_output)

    def test_sequence_path_input(self):
        input_paths = [Path("/path/to/file1.jpg"), Path("/path/to/file2.jpg")]
        expected_output = [Path("/path/to/file1.jpg"), Path("/path/to/file2.jpg")]
        self.assertEqual(clean_input_paths(input_paths), expected_output)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            clean_input_paths("")

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            clean_input_paths(123)

    def test_mixed_type_input(self):
        mixed_input = [Path("/path/to/file1.jpg"), "/path/to/file2.jpg", 123]
        with self.assertRaises(TypeError):
            clean_input_paths(mixed_input)


if __name__ == "__main__":
    unittest.main()
