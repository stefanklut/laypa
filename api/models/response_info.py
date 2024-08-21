# Imports

# > Standard Libraries
from typing import TypedDict


class ResponseInfo(TypedDict, total=False):
    status_code: int
    identifier: str
    filename: str
    whitelist: list[str]
    added_queue_position: int
    remaining_queue_size: int
    added_time: str
    model_name: str
    error_message: str
