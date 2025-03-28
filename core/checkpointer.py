import logging
from pathlib import Path
from typing import Any, override

from detectron2.checkpoint import DetectionCheckpointer


class NewDetectionCheckpointer(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)

    # TODO Write the override methods for save and load
    @override
    def save(self, name: str, **kwargs: Any) -> None:
        return super().save(name, **kwargs)

    @override
    def load(self, path, *args, **kwargs):
        return super().load(path, *args, **kwargs)
