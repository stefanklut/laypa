# Imports

# > Standard Libraries
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# > External Libraries
from prometheus_client import Gauge, Counter


@dataclass
class DummyArgs:
    """
    Args to be used instead of the argparse.Namespace
    """

    config: str = "config.yaml"
    output: str = ""
    opts: list[str] = field(default_factory=list)


def initialize_environment(max_queue_size, output_base_path):
    args = DummyArgs(output=str(output_base_path))
    max_workers = 1
    max_queue_size = max_workers + max_queue_size

    # Run a separate thread on which the GPU runs and processes requests put in
    # the queue
    executor = ThreadPoolExecutor(max_workers=max_workers)

    # Prometheus metrics to be returned
    queue_size_gauge = Gauge("queue_size", "Size of worker queue").set_function(lambda: executor._work_queue.qsize())
    images_processed_counter = Counter("images_processed", "Total number of images processed")
    exception_predict_counter = Counter("exception_predict", "Exception thrown in predict() function")

    return (args, executor, queue_size_gauge, images_processed_counter, exception_predict_counter)
