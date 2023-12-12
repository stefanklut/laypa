import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.input_utils import clean_input_paths, get_file_paths


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

    def test_empty_input_paths(self):
        input_path = None
        formats = {".jpg", ".png"}
        with self.assertRaises(TypeError):
            get_file_paths(input_path, formats)

    def test_empty_formats(self):
        input_path = "/path/to/single/file.jpg"
        formats = set()
        with self.assertRaises(ValueError):
            get_file_paths(input_path, formats)

    def test_nonexistent_input_path(self):
        input_path = "/nonexistent/path"
        formats = {".jpg", ".png"}
        with self.assertRaises(FileNotFoundError):
            get_file_paths(input_path, formats)

    def test_permission_error_input_path(self):
        input_path = "/protected/path"
        formats = {".jpg", ".png"}
        with self.assertRaises(PermissionError):
            get_file_paths(input_path, formats)

    def test_invalid_file_type(self):
        input_path = "/path/to/file.txt"
        formats = {".jpg", ".png"}
        with self.assertRaises(ValueError):
            get_file_paths(input_path, formats)

    def test_directory_with_no_supported_files(self):
        input_path = "/path/to/empty/directory"
        formats = {".jpg", ".png"}
        with self.assertRaises(FileNotFoundError):
            get_file_paths(input_path, formats)

    def test_txt_file_with_nonexistent_paths(self):
        input_path = "/path/to/invalid/files.txt"
        formats = {".jpg", ".png"}
        with self.assertRaises(FileNotFoundError):
            get_file_paths(input_path, formats)


if __name__ == "__main__":
    unittest.main()
