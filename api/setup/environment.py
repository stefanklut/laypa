# Imports

# > Standard Libraries
import os
from pathlib import Path


def read_environment_variables():
    try:
        max_queue_size_string: str = os.environ["LAYPA_MAX_QUEUE_SIZE"]
        model_base_path_string: str = os.environ["LAYPA_MODEL_BASE_PATH"]
        output_base_path_string: str = os.environ["LAYPA_OUTPUT_BASE_PATH"]
    except KeyError as error:
        raise KeyError(f"Missing Laypa Environment variable: {error.args[0]}")

    # Convert to appropriate types
    max_queue_size = int(max_queue_size_string)
    model_base_path = Path(model_base_path_string)
    output_base_path = Path(output_base_path_string)

    # Check if paths exist
    if not model_base_path.is_dir():
        raise FileNotFoundError(
            f"LAYPA_MODEL_BASE_PATH: {model_base_path} is not found in the current filesystem")
    if not output_base_path.is_dir():
        raise FileNotFoundError(
            f"LAYPA_OUTPUT_BASE_PATH: {output_base_path} is not found in the current filesystem")

    return max_queue_size, model_base_path, output_base_path
